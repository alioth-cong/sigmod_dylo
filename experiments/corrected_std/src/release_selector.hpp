#pragma once
#include <chrono>
#include "corrected_state.hpp"
#include "closure.hpp"
#include "localized_closure.hpp"
namespace dylo_corrected {
struct SelectorStats {std::string solver;double ms=0;size_t candidates=0,eligible=0,pruned=0;double objective=0;};
template<class Tree> dylo::LocalizedSelection release_select(const Tree& t,const std::vector<dylo::AtomicAction>& actions,const Ledger& ledger,double lambda,const std::string& mode,bool use_dp,bool prune,IncrementalDP& dp,bool verify,SelectorStats& stats){
  auto started=std::chrono::steady_clock::now();
  dylo::LocalizedSelection out;out.selected_actions.assign(actions.size(),0);out.dirty_nodes.assign(t.parent.size(),0);
  dp.repaired=0;stats.candidates=actions.size();bool single=true;std::map<Key,double> gain;
  std::vector<char> eligible(actions.size(),0),child(t.parent.size(),0);
  for(int p:t.parent)if(p>=0)child[p]=1;
  for(size_t i=0;i<actions.size();i++)if(actions[i].gain>0){eligible[i]=1;single&=actions[i].anchors.size()==1;if(actions[i].anchors.size()==1)gain[t.nodeKey.at(actions[i].anchors[0])]+=actions[i].gain;}
  if(prune&&single){std::set<Key> removed;for(auto [key,g]:gain){int n=t.ids.at(key.first).at(key.second);if(!child[n]&&g<=lambda*ledger.marginal(key))removed.insert(key);}
    for(size_t i=0;i<actions.size();i++)if(eligible[i]&&removed.count(t.nodeKey[actions[i].anchors[0]])){eligible[i]=0;stats.pruned++;}for(Key key:removed)gain.erase(key);}
  stats.eligible=std::count(eligible.begin(),eligible.end(),char(1));
  const bool exact=mode=="dp"||mode=="cut";
  stats.solver=exact?(mode=="cut"||!use_dp||!single?"cut":"dp"):mode;
  if(stats.eligible){
    if(stats.solver=="dp"){
      auto opened=dp.solve(t,gain,ledger,lambda,verify);
      for(size_t i=0;i<actions.size();i++)if(eligible[i]&&opened.count(t.nodeKey[actions[i].anchors[0]]))out.selected_actions[i]=1;
    }else if(stats.solver=="cut"){
      std::vector<dylo::AtomicAction> compact;std::vector<size_t> mapping;std::vector<double> weights(t.parent.size());
      for(size_t n=0;n<weights.size();n++)weights[n]=ledger.marginal(t.nodeKey[n]);
      for(size_t i=0;i<actions.size();i++)if(eligible[i]){compact.push_back(actions[i]);mapping.push_back(i);}
      auto chosen=dylo::select_atomic_actions(t.parent,weights,compact,lambda);
      for(size_t j=0;j<mapping.size();j++)out.selected_actions[mapping[j]]=chosen.selected_actions[j];
    }else{
      std::map<int,std::vector<size_t>> buckets;
      for(size_t i=0;i<actions.size();i++)if(eligible[i]){
        int bucket=mode=="indep"?int(i):(actions[i].destination>=0?actions[i].destination:actions[i].anchors.at(0));
        buckets[bucket].push_back(i);
      }
      for(auto& [bucket,indices]:buckets){
        double g=0,w=0;std::set<int> path;
        for(size_t i:indices){g+=actions[i].gain;for(int n:actions[i].anchors)for(int c=n;c>=0;c=t.parent[c])path.insert(c);}
        for(int c:path)w+=ledger.marginal(t.nodeKey[c]);
        if(g>lambda*w+1e-12)for(size_t i:indices)out.selected_actions[i]=1;
      }
    }
    for(size_t i=0;i<actions.size();i++)if(out.selected_actions[i]){out.selected_gain+=actions[i].gain;for(int n:actions[i].anchors)for(int c=n;c>=0;c=t.parent[c])out.dirty_nodes[c]=1;}
    for(size_t n=0;n<out.dirty_nodes.size();n++)if(out.dirty_nodes[n])out.marginal_cost+=ledger.marginal(t.nodeKey[n]);
  }
  out.objective=out.selected_gain-lambda*out.marginal_cost;stats.objective=out.objective;
  stats.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return out;
}
} // namespace
