#pragma once
#include "detector_fixed.hpp"
#include <map>
#include <stdexcept>
namespace dylo {
// Detector iterations may redistribute refined subcommunities and their flat
// partitions need not nest. An index tree must nest. Intersect each intermediate
// partition with its already repaired parent, preserving the fixed leaf and top
// partitions, and preserve existing labels wherever no split is necessary.
inline size_t nest_release_hierarchy(Hierarchy& H,const Graph& g){
  size_t split=0;
  for(int li=int(H.levels.size())-2;li>=1;--li){
    auto& level=H.levels[li];const auto& above=H.levels[li+1];
    int next=0;for(int c:level)next=std::max(next,c+1);
    std::map<std::pair<int,int>,int> ids;std::map<int,int> first;
    for(int v=0;v<g.n;v++)if(g.k[v]>0){
      int original=level.at(v),parent=above.at(v);auto key=std::make_pair(original,parent);
      auto it=ids.find(key);
      if(it==ids.end()){
        int id=original;
        if(first.count(original)){id=next++;split++;}else first[original]=parent;
        it=ids.emplace(key,id).first;
      }
      level[v]=it->second;
    }
  }
  for(size_t li=0;li+1<H.levels.size();li++){
    std::map<int,int> parent;
    for(int v=0;v<g.n;v++)if(g.k[v]>0){
      int c=H.levels[li].at(v),p=H.levels[li+1].at(v);
      auto [it,added]=parent.emplace(c,p);
      if(!added&&it->second!=p)throw std::runtime_error("fixed summary leaf crosses parent partition");
    }
  }
  return split;
}
inline Hierarchy build_release_levels_from_leaf(const Graph& g,std::vector<int> leaf,Instance inst,std::mt19937& rng,double gamma=1){
  auto H=build_levels_from_leaf(g,std::move(leaf),inst,rng,gamma);nest_release_hierarchy(H,g);return H;
}
inline Hierarchy build_release_levels_combined(const Graph& g,std::vector<int> coarse,Instance inst,std::mt19937& rng,int maxSize=25,double gamma=.5,int maxDepth=6){
  auto H=build_levels_combined(g,std::move(coarse),inst,rng,maxSize,gamma,maxDepth);nest_release_hierarchy(H,g);return H;
}
} // namespace
