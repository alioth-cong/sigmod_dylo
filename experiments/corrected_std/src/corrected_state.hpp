#pragma once
#include <map>
#include <string>
#include <set>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace dylo_corrected {
using Key=std::pair<int,int>;
struct Share {Key node; int source; double amount;};
struct Quote {
  std::set<Key> open;
  std::vector<Share> book;
  std::set<Key> cancel;
  double execution=0,retirement=0,booking=0,cancellation=0,delta=0;
};
struct Valuation {double weight=0; std::set<int> owners; unsigned long version=0;};

// A report version freezes its weight and ownership until it is materialized.
// Currency is raw structural units. Normalizing by the initial full-index
// weight is optional at the caller; the ledger always conserves raw currency.
class Ledger {
 public:
  std::map<Key,Valuation> values;
  std::map<Key,std::map<int,double>> pending;
  std::map<Key,double> opened;
  std::map<Key,std::map<int,double>> retired_this_batch;
  std::map<Key,std::string> reasons;
  double booked=0,retired=0,canceled=0,total_executed=0;
  double initial_scale=1,batch_pending_start=0;
  unsigned long next_version=1;
  double balance()const {double s=0;for(const auto& [k,p]:pending)for(const auto& [a,x]:p)s+=x;return s;}
  double pending_at(Key k)const {auto p=pending.find(k);double s=0;if(p!=pending.end())for(auto [a,x]:p->second)s+=x;return s;}
  double weight(Key k)const {auto p=values.find(k);return p==values.end()?0:p->second.weight;}
  double marginal(Key k)const {return opened.count(k)?0:weight(k);}
  double owned(Key k,int a)const {
    auto p=values.find(k);if(p==values.end()||!p->second.owners.count(a))return 0;
    return p->second.weight/std::max<size_t>(1,p->second.owners.size());
  }
  double unbooked(Key k,int a)const {
    if(opened.count(k))return 0;
    double x=0;auto p=pending.find(k);if(p!=pending.end()){auto q=p->second.find(a);if(q!=p->second.end())x=q->second;}
    return std::max(0.0,owned(k,a)-x);
  }
  template<class Tree> void ensure(const Tree& t,bool replace=false) {
    if(t.ids.empty())return;
    std::map<Key,std::set<int>> owners;
    for(const auto& [a,n]:t.ids[0])
      for(int c=n;c>=0;c=t.parent[c])owners[t.nodeKey[c]].insert(a);
    for(size_t c=0;c<t.nodeKey.size();c++){
      Key k=t.nodeKey[c];
      if(t.weight[c]<=0||!t.ids[k.first].count(k.second))continue;
      if(replace||!values.count(k))values[k]={t.weight[c],owners[k],next_version++};
    }
  }
  void begin(){opened.clear();retired_this_batch.clear();reasons.clear();batch_pending_start=balance();}
  Quote quote(const std::set<Key>& paths,const std::vector<Share>& shares,const std::set<Key>& gone={})const {
    Quote q;q.cancel=gone;
    for(Key k:paths)if(!opened.count(k)&&!gone.count(k)){
      q.open.insert(k);q.execution+=weight(k);q.retirement+=pending_at(k);
    }
    for(Key k:gone)if(!q.open.count(k))q.cancellation+=pending_at(k);
    std::map<std::pair<Key,int>,double> sums;
    for(const auto& s:shares)if(!paths.count(s.node)&&!opened.count(s.node)&&!gone.count(s.node))sums[{s.node,s.source}]+=s.amount;
    for(auto [id,x]:sums){double b=std::min(x,unbooked(id.first,id.second));if(b>1e-14){q.book.push_back({id.first,id.second,b});q.booking+=b;}}
    q.delta=q.execution-q.retirement+q.booking-q.cancellation;
    return q;
  }
  void commit(const Quote& q,const std::string& reason="move"){
    for(Key k:q.open){
      opened[k]=weight(k);reasons[k]=reason;
      auto p=pending.find(k);if(p!=pending.end()){retired_this_batch[k]=p->second;for(auto [a,x]:p->second)retired+=x;pending.erase(p);}
    }
    for(Key k:q.cancel){auto p=pending.find(k);if(p!=pending.end()){for(auto [a,x]:p->second)canceled+=x;pending.erase(p);}}
    for(const auto& s:q.book){pending[s.node][s.source]+=s.amount;booked+=s.amount;}
    check();
  }
  void schedule(const std::set<Key>& paths,const std::string& reason){commit(quote(paths,{}),reason);}
  void check()const {
    double p=balance();if(std::fabs(p-(booked-retired-canceled))>1e-7*std::max(1.0,booked))throw std::runtime_error("ledger conservation failure");
    for(const auto& [k,ps]:pending){double s=0;for(auto [a,x]:ps){s+=x;if(x<0||x>owned(k,a)+1e-7)throw std::runtime_error("ownership overflow");}if(s>weight(k)+1e-7)throw std::runtime_error("node liability overflow");}
  }
  template<class Tree> double finish(const Tree& t){
    std::set<Key> live;for(size_t c=0;c<t.nodeKey.size();c++){Key k=t.nodeKey[c];if(t.weight[c]>0&&t.ids[k.first].count(k.second))live.insert(k);}
    for(auto p=pending.begin();p!=pending.end();){if(!live.count(p->first)){for(auto [a,x]:p->second)canceled+=x;p=pending.erase(p);}else ++p;}
    double executed=0;
    for(auto p=opened.begin();p!=opened.end();){
      if(!live.count(p->first)){
        for(auto [a,x]:retired_this_batch[p->first]){retired-=x;canceled+=x;}
        retired_this_batch.erase(p->first);reasons.erase(p->first);p=opened.erase(p);
      }else{executed+=p->second;++p;}
    }
    total_executed+=executed;
    // Refreshed nodes start a new report valuation/ownership epoch. Nodes
    // with deferred reports retain their old epoch even when sizes change.
    std::map<Key,std::set<int>> owners;
    if(!t.ids.empty())for(const auto& [a,n]:t.ids[0])for(int c=n;c>=0;c=t.parent[c])if(opened.count(t.nodeKey[c]))owners[t.nodeKey[c]].insert(a);
    for(size_t c=0;c<t.nodeKey.size();c++)if(opened.count(t.nodeKey[c]))values[t.nodeKey[c]]={t.weight[c],owners[t.nodeKey[c]],next_version++};
    for(auto p=values.begin();p!=values.end();)if(!live.count(p->first))p=values.erase(p);else ++p;
    check();return executed;
  }
  template<class Tree> void rebuild(const Tree& t){
    for(auto& [k,p]:pending)for(auto [a,x]:p)canceled+=x;
    pending.clear();
    for(auto& [k,p]:retired_this_batch)for(auto [a,x]:p){retired-=x;canceled+=x;}
    opened.clear();retired_this_batch.clear();values.clear();reasons.clear();
    ensure(t);std::set<Key> all;for(const auto& [k,v]:values)all.insert(k);schedule(all,"hierarchy_rebuild");
  }
};

inline double relative_drift(double now,double ref){if(ref==0)return now==0?0:std::numeric_limits<double>::infinity();return std::fabs(now-ref)/std::fabs(ref);}

// Persistent DP values. Topology/weights are reconciled against the patched
// cost tree; only changed local values and their ancestor contributions are
// repaired. A fresh recurrence is available for independent verification.
class IncrementalDP {
  struct Node {double gain=0,cost=0,val=0,pos_children=0;};
  std::vector<Key> keys;
  std::vector<int> parent,order;
  std::vector<char> active;
  std::vector<std::vector<int>> children;
  std::vector<Node> nodes;
 public:
  size_t repaired=0;
  template<class Tree> std::set<Key> solve(const Tree& t,const std::map<Key,double>& gain,const Ledger& ledger,double lambda,bool verify=false){
    size_t n=t.nodeKey.size();std::vector<char> live(n,0);
    for(size_t c=0;c<n;c++){
      Key k=t.nodeKey[c];
      if(k.first>=0&&size_t(k.first)<t.ids.size()){
        auto id=t.ids[k.first].find(k.second);live[c]=id!=t.ids[k.first].end()&&id->second==int(c);
      }
    }
    if(keys!=t.nodeKey||parent!=t.parent||active!=live){
      keys=t.nodeKey;parent=t.parent;active=live;nodes.assign(n,Node{});children.assign(n,{});order.clear();
      std::vector<std::vector<int>> levels(t.ids.size());
      for(size_t c=0;c<n;c++)if(active[c]){
        levels.at(keys[c].first).push_back(c);
        int p=parent[c];
        if(p>=0){
          if(size_t(p)>=n||!active[p]||keys[p].first<=keys[c].first)throw std::runtime_error("invalid DP parent dependency");
          children[p].push_back(c);
        }
      }
      for(auto& level:levels)order.insert(order.end(),level.begin(),level.end());
      // Stable summation preserves replay across tree rebuilds with different IDs.
      for(auto& cs:children)std::sort(cs.begin(),cs.end(),[&](int a,int b){return keys[a]<keys[b];});
    }
    std::vector<double> gains(n,0);std::vector<char> dirty(n,0);
    for(auto [key,g]:gain){
      if(key.first<0||size_t(key.first)>=t.ids.size())throw std::runtime_error("gain outside DP tree");
      auto id=t.ids[key.first].find(key.second);
      if(id==t.ids[key.first].end())throw std::runtime_error("gain outside active DP tree");
      gains[id->second]=g;
    }
    for(int c:order){
      auto& node=nodes[c];double cost=lambda*ledger.marginal(keys[c]);
      dirty[c]=node.gain!=gains[c]||node.cost!=cost;node.gain=gains[c];node.cost=cost;
      node.pos_children=0;for(int ch:children[c])node.pos_children+=std::max(nodes[ch].val,0.0);
    }
    repaired=0;
    for(int c:order)if(dirty[c]){
      auto& node=nodes[c];double old=std::max(node.val,0.0);node.val=node.gain-node.cost+node.pos_children;repaired++;
      double delta=std::max(node.val,0.0)-old;
      if(delta!=0&&parent[c]>=0){nodes[parent[c]].pos_children+=delta;dirty[parent[c]]=1;}
    }
    if(verify){
      std::vector<double> values(n,0);
      for(int c:order){double value=nodes[c].gain-nodes[c].cost;for(int ch:children[c])value+=std::max(values[ch],0.0);values[c]=value;
        if(std::fabs(value-nodes[c].val)>1e-8*std::max(1.0,std::fabs(value)))throw std::runtime_error("incremental DP mismatch");}
    }
    std::set<Key> out;std::vector<int> stack;
    for(int c:order)if(parent[c]<0&&nodes[c].val>0)stack.push_back(c);
    while(!stack.empty()){int c=stack.back();stack.pop_back();out.insert(keys[c]);for(int ch:children[c])if(nodes[ch].val>0)stack.push_back(ch);}
    return out;
  }
  void clear(){keys.clear();parent.clear();order.clear();active.clear();children.clear();nodes.clear();}
};

// Exact group modularity change in the original driver's m*DeltaQ units.
// Every edge incident to a mover is compared before/after, once, including
// edges between two movers. Self-loop terms cancel under reassignment.
template<class Graph> double exact_group_gain(const Graph& g,const std::vector<int>& before,const std::vector<int>& after,const std::vector<int>& movers,const std::unordered_map<int,double>& totals){
  std::unordered_set<int> moving(movers.begin(),movers.end());
  std::unordered_map<int,double> delta;
  double internal=0;
  for(int u:movers){delta[before[u]]-=g.k[u];delta[after[u]]+=g.k[u];
    for(auto [v,w]:g.adj[u]){if((int)v==u||(moving.count(v)&&u>(int)v))continue;
      internal+=w*((after[u]==after[v]?1:0)-(before[u]==before[v]?1:0));}}
  double null_delta=0;
  for(auto [c,d]:delta){auto p=totals.find(c);double x=p==totals.end()?0:p->second;null_delta+=(x+d)*(x+d)-x*x;}
  return internal-null_delta/(2*g.m2);
}
} // namespace dylo_corrected
