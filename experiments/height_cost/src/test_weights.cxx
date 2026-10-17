#define main dylo_driver_main
#include "driver_height.cxx"
#undef main
#include <cassert>
int main(){
  Graph g;g.ensure(6);
  for(int i=0;i<5;i++)g.add_edge(i,i+1,1);
  Hierarchy H;H.levels={{0,0,1,1,2,2},{0,0,0,0,1,1},{0,0,0,0,0,0}};
  auto t=make_cost_tree(g,H);
  auto &m=wmodel();
  assert(t.weight.size()==6);
  double total=0;
  for(double w:t.weight){assert(std::isfinite(w)&&w>0);total+=w;}
  assert(std::abs(total-11.0)<1e-9); // 6 entities + 3 first-level children + 2 top children
  const int root=t.ids[2].at(0), p=t.ids[1].at(0), leaf=t.ids[0].at(0);
  if(m.height){
    auto expected=[&](int h,double n,double count){auto row=m.layers[h];return m.S*std::min(row[0],row[1]+row[2]*(m.structural?count:n));};
    assert(std::abs(t.weight[root]-expected(2,6,2))<1e-9);
    assert(std::abs(t.weight[p]-expected(1,4,2))<1e-9);
    assert(std::abs(t.weight[leaf]-expected(0,2,2))<1e-9);
    double scale=m.S;
    H.levels[0][1]=1;
    make_cost_tree(g,H);
    assert(m.S==scale); // scale is fixed, never refitted after an update
    assert(node_cost_of_size(1)>0);
    assert(node_cost_of_size(1)<=node_cost_of_size(10));
    assert(node_cost_of_size(10000)==node_cost_of_size(100000));
  }else{assert(t.weight[root]==2&&t.weight[p]==2&&t.weight[leaf]==2);}
  puts("height, subtree size, saturation, positive weights and fixed scaling: PASS");
}
