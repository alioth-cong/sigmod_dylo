#include "src/release_hierarchy.hpp"
#include <cassert>
#include <iostream>
int main(){using namespace dylo;Graph g;for(int i=0;i<8;i++)g.add_edge(i,(i+1)%8,1);
 Hierarchy h;h.levels={{0,0,1,1,2,2,3,3},{10,10,10,10,11,11,11,11},{20,20,21,21,20,20,21,21}};
 auto leaf=h.levels.front(),top=h.levels.back();assert(nest_release_hierarchy(h,g)==2);assert(h.levels.front()==leaf&&h.levels.back()==top);assert(nest_release_hierarchy(h,g)==0);
 h.levels[1][1]=999;bool rejected=false;try{nest_release_hierarchy(h,g);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 std::cout<<"PASS summary hierarchy: split crossing intermediate nodes, fixed leaves/top preserved, idempotent, invalid leaf dependency rejected\n";
}
