// DyLo Algorithm 2 — OpenSet exact DP over the affected hierarchy.
// Faithful C++ port of harness/graphrag/dylo_dp_prototype.py (validated by its selftest).
// Returns the ancestor-closed dirty set D over T_A maximizing  sum_{c in D} (g(c) - lambda*w(c)).
//   open(c) = g(c) - lambda*w(c) + sum_{c' in children(c)} max(open(c'), 0)
//   root dirty iff open(root) >= 0; pre-order: a child is dirty iff open(child) >= 0.
//   A community c is actually re-summarized (in L) iff it is dirty AND g(c) > 0.
#pragma once
#include <vector>
#include <cstdint>

namespace dylo {

// Tree given by children adjacency (node ids 0..N-1); roots = nodes with parent<0.
// g[c], w[c] per node. Fills `dirty` (ancestor-closed committed set) and returns it.
// Multi-root supported (a forest): each root decided independently.
struct DPTree {
  std::vector<std::vector<int>> children;   // children[c] = child ids
  std::vector<int> roots;                    // root ids
  std::vector<double> g, w;                  // per-node gain / cost weight
};

// Post-order via explicit stack; then pre-order extraction. dirty[c]=1 => committed.
inline std::vector<char> open_set_dp(const DPTree& T, double lambda){
  int N = (int)T.g.size();
  std::vector<double> op(N, 0.0);
  std::vector<char> dirty(N, 0);
  // post-order over the forest
  std::vector<int> order; order.reserve(N);
  std::vector<int> stack(T.roots.begin(), T.roots.end());
  while(!stack.empty()){
    int c = stack.back(); stack.pop_back(); order.push_back(c);
    for(int ch : T.children[c]) stack.push_back(ch);
  }
  // reverse -> children before parents
  for(int i=(int)order.size()-1;i>=0;i--){
    int c = order[i];
    double s = T.g[c] - lambda*T.w[c];
    for(int ch : T.children[c]) if(op[ch] > 0.0) s += op[ch];
    op[c] = s;
  }
  // pre-order extraction: dirty a root iff op>=0, then descend
  std::vector<int> st;
  for(int r : T.roots) if(op[r] >= 0.0){ dirty[r]=1; st.push_back(r); }
  while(!st.empty()){
    int c = st.back(); st.pop_back();
    for(int ch : T.children[c]) if(op[ch] >= 0.0){ dirty[ch]=1; st.push_back(ch); }
  }
  return dirty;
}

} // namespace dylo
