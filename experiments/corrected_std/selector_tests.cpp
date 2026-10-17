#include "src/release_selector.hpp"
#include <random>
#include <iostream>
#include <cassert>
using namespace dylo;using namespace dylo_corrected;
struct Tree{std::vector<int> parent;std::vector<double> weight;std::vector<Key> nodeKey;std::vector<std::unordered_map<int,int>> ids;};
int main(){std::mt19937 rng(20261014);int solved=0;
 for(int test=0;test<4000;test++){
  int n=1+rng()%9,na=1+rng()%8;Tree t;t.parent.resize(n);t.weight.resize(n);t.nodeKey.resize(n);
  std::vector<int> heights(n,0);for(int i=0;i<n;i++)t.parent[i]=(i&&rng()%4)?int(rng()%i):-1;
  for(int i=n-1;i>0;i--)if(t.parent[i]>=0)heights[t.parent[i]]=std::max(heights[t.parent[i]],heights[i]+1);
  t.ids.resize(1+*std::max_element(heights.begin(),heights.end()));Ledger l;
  for(int i=0;i<n;i++){t.weight[i]=rng()%6;t.nodeKey[i]={heights[i],i};t.ids[heights[i]][i]=i;l.values[t.nodeKey[i]]={t.weight[i],{},0};if(rng()%5==0)l.opened[t.nodeKey[i]]=t.weight[i];}
  std::vector<AtomicAction> actions;bool single=test%2==0;
  for(int i=0;i<na;i++){AtomicAction a;a.gain=(int(rng()%25)-5)*.25;a.anchors={int(rng()%n)};if(!single){a.anchors.push_back(rng()%n);std::sort(a.anchors.begin(),a.anchors.end());a.anchors.erase(std::unique(a.anchors.begin(),a.anchors.end()),a.anchors.end());}actions.push_back(a);}
  double lambda=(rng()%6)*.25,best=0;
  for(int mask=0;mask<(1<<na);mask++){double gain=0,cost=0;std::set<int> opened;
   for(int i=0;i<na;i++)if(mask&(1<<i)){gain+=actions[i].gain;for(int anchor:actions[i].anchors)for(int c=anchor;c>=0;c=t.parent[c])opened.insert(c);}
   for(int c:opened)cost+=l.marginal(t.nodeKey[c]);best=std::max(best,gain-lambda*cost);}
  IncrementalDP cache;SelectorStats stats;
  for(std::string mode:std::vector<std::string>{"dp","cut","greedy","indep"})for(bool prune:{false,true}){
   auto out=release_select(t,actions,l,lambda,mode,true,prune,cache,true,stats);
   if(mode=="dp"||mode=="cut")assert(std::fabs(out.objective-best)<1e-8);else assert(out.objective<=best+1e-8);
   for(size_t i=0;i<actions.size();i++)if(out.selected_actions[i])assert(actions[i].gain>0);
   solved++;
  }
 }
 std::cout<<"PASS release selectors/enumeration="<<solved<<"\n";
}
