#include "src/corrected_state.hpp"
#include "src/closure.hpp"
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cassert>
using namespace dylo;using namespace dylo_corrected;
struct Tree{std::vector<int> parent;std::vector<double> weight;std::vector<Key> nodeKey;std::vector<std::unordered_map<int,int>> ids;};
struct Case{Tree t;Ledger l;std::map<Key,double> gain;std::vector<double> net;std::vector<AtomicAction> actions;std::vector<double> cut_weight;double lambda;};
Case make_case(int n,std::mt19937& rng){Case c;c.lambda=std::vector<double>{0,.1,.5,1,2}[rng()%5];c.t.parent.resize(n);std::vector<int> heights(n,0);c.net.resize(n);c.cut_weight.resize(n);
 for(int i=0;i<n;i++)c.t.parent[i]=i&&rng()%10?int(rng()%i):-1;
 for(int i=n-1;i>0;i--)if(c.t.parent[i]>=0)heights[c.t.parent[i]]=std::max(heights[c.t.parent[i]],heights[i]+1);
 c.t.ids.resize(1+*std::max_element(heights.begin(),heights.end()));
 for(int i=0;i<n;i++){double w=1+rng()%6,g=std::uniform_real_distribution<double>(-10,10)(rng);Key k={heights[i],i};c.t.weight.push_back(w);c.t.nodeKey.push_back(k);c.t.ids[k.first][i]=i;c.l.values[k]={w,{},0};c.gain[k]=g;c.net[i]=g-c.lambda*w;c.cut_weight[i]=std::max(-c.net[i],0.0);if(c.net[i]>0)c.actions.push_back({c.net[i],{i}});}
 return c;
}
double objective(const Case& c,const std::set<Key>& opened){double value=0;for(int i=0;i<(int)c.net.size();i++)if(opened.count(c.t.nodeKey[i]))value+=c.net[i];return value;}
double brute(const Case& c){int n=c.net.size();std::vector<unsigned> anc(n),closed(1u<<n);std::vector<double> value(1u<<n);for(int i=0;i<n;i++)for(int p=i;p>=0;p=c.t.parent[p])anc[i]|=1u<<p;double best=0;
 for(unsigned m=1;m<(1u<<n);m++){int i=__builtin_ctz(m);unsigned prior=m&(m-1);closed[m]=closed[prior]|anc[i];value[m]=value[prior]+c.net[i];if(closed[m]==m)best=std::max(best,value[m]);}return best;}
int main(int argc,char** argv){std::mt19937 rng(20261014);for(int test=0;test<5000;test++){int n=test<80?14+rng()%9:1+rng()%14;auto c=make_case(n,rng);IncrementalDP dp;auto opened=dp.solve(c.t,c.gain,c.l,c.lambda,true);auto cut=select_atomic_actions(c.t.parent,c.cut_weight,c.actions,1);double a=objective(c,opened),b=brute(c);assert(std::fabs(a-b)<1e-8&&std::fabs(cut.objective-b)<1e-8);}
 std::cout<<"PASS signed-node forests: exhaustive/DP/closure=5000, max nodes=22\n";
 std::ofstream out(argc>1?argv[1]:"selector_scaling.csv");out<<"nodes,repeat,dp_ms,cut_ms,objective\n";
 for(int n:{100,500,1000,5000,10000,20000})for(int repeat=0;repeat<10;repeat++){auto c=make_case(n,rng);IncrementalDP dp;auto start=std::chrono::steady_clock::now();auto opened=dp.solve(c.t,c.gain,c.l,c.lambda);auto middle=std::chrono::steady_clock::now();auto cut=select_atomic_actions(c.t.parent,c.cut_weight,c.actions,1);auto end=std::chrono::steady_clock::now();double value=objective(c,opened);assert(std::fabs(value-cut.objective)<1e-7*std::max(1.0,std::fabs(value)));out<<n<<','<<repeat<<','<<std::chrono::duration<double,std::milli>(middle-start).count()<<','<<std::chrono::duration<double,std::milli>(end-middle).count()<<','<<value<<'\n';}
 std::cout<<"PASS scaling DP/closure=60, up to 20000 nodes\n";
}
