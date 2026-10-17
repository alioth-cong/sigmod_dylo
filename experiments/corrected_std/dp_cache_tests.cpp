#include "src/release_selector.hpp"
#include <random>
#include <cassert>
#include <iostream>
using namespace dylo;using namespace dylo_corrected;
struct Tree{std::vector<int> parent;std::vector<double> weight;std::vector<Key> nodeKey;std::vector<std::unordered_map<int,int>> ids;};
int main(){std::mt19937 rng(20261014);int cases=0;
 for(int test=0;test<500;test++){IncrementalDP cache;
  for(int step=0;step<12;step++){
   int n=1+rng()%30;Tree t;t.parent.resize(n);t.weight.resize(n);t.nodeKey.resize(n);std::vector<int> heights(n,0);
   for(int i=0;i<n;i++)t.parent[i]=i&&rng()%4?int(rng()%i):-1;
   for(int i=n-1;i>0;i--)if(t.parent[i]>=0)heights[t.parent[i]]=std::max(heights[t.parent[i]],heights[i]+1);
   t.ids.resize(1+*std::max_element(heights.begin(),heights.end()));Ledger l;std::map<Key,double> gains;std::vector<AtomicAction> actions;
   for(int i=0;i<n;i++){Key k={heights[i],i};t.nodeKey[i]=k;t.weight[i]=rng()%6;t.ids[k.first][i]=i;l.values[k]={t.weight[i],{},0};if(rng()%4==0)l.opened[k]=t.weight[i];double g=(rng()%21)*.25;gains[k]=g;if(g>0)actions.push_back({g,{i}});}
   // Multiple numerical updates on the same tree exercise propagation with
   // unchanged topology; the next step changes topology, IDs and node count.
   for(int round=0;round<3;round++){
    double lambda=(rng()%6)*.25;std::vector<double> weights(n);for(int i=0;i<n;i++)weights[i]=l.marginal(t.nodeKey[i]);
    auto opened=cache.solve(t,gains,l,lambda,true);auto cut=select_atomic_actions(t.parent,weights,actions,lambda);double value=0;
    for(Key k:opened)value+=gains[k]-lambda*l.marginal(k);
    assert(std::fabs(value-cut.objective)<1e-8);cases++;
    int i=rng()%n;l.opened[t.nodeKey[i]]=t.weight[i];
   }
  }
 }
 std::cout<<"PASS dense DP cache: changing topology/IDs/count, free and prepaid nodes, repeated price/rate changes="<<cases<<"\n";
}
