// Localized (kernelized) exact multi-anchor selection: an adaptive layer on
// top of closure.hpp's exact max-weight-closure solver.
//
// Buckets candidate actions by LCA in the tree. Two buckets h,k are
// independently solvable iff neither's below-LCA private cost region L_h
// intersects the OTHER bucket's full payment surface (L_k union Anc(k),
// noting Anc(k) includes k itself) -- this is the corrected decomposability
// certificate (a bucket-vs-bucket lower/lower check alone is NOT sufficient:
// a bucket's private region can contain another bucket's own anchor
// position, coupling them even with disjoint "lower" sets literally).
// Singleton (uncoupled) buckets solve a SMALL local min-cut and compress to
// one super-action at position h. Coupled buckets (found via union-find over
// per-node ownership, so transitively-linked buckets end up in one
// component) keep their real, uncompressed actions. Everything retained
// (super-actions + coupled buckets' real actions) is combined into ONE final
// min-cut over a network PRUNED to just the touched ancestor nodes -- not
// the original single-anchor tree DP for the "clean" part, because a clean
// bucket's ancestor path can still share cost with a coupled component's own
// payment surface; only a single joint optimization prices that correctly.
//
// Exact whenever every bucket ends up in a singleton component (verified
// zero objective mismatch against the global solver on real 3-dataset runs,
// see harness/cpp/dylo/localized_closure_test.cxx); safely falls back to
// (kernelized, not full-global) min-cut for coupled buckets otherwise, so
// correctness never regresses below select_atomic_actions.
#pragma once

#include "closure.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace dylo {

struct LocalizedSelection {
  std::vector<char> selected_actions;  // indexed like the input `actions`
  std::vector<char> dirty_nodes;       // indexed like `parent`/`weight`
  double selected_gain = 0.0;
  double marginal_cost = 0.0;
  double objective = 0.0;
  long network_nodes = 0;   // summed across every flow network actually built
  long network_edges = 0;   // (the real cost driver of Dinic's running time)
};

namespace localized_detail {

inline std::vector<int> ancestors_incl(const std::vector<int>& parent, int c) {
  std::vector<int> path;
  while (c >= 0) {
    path.push_back(c);
    c = parent[c];
  }
  return path;
}

inline int lca_pair(const std::vector<int>& parent, const std::vector<int>& depth,
                     int a, int b) {
  // O(depth) two-pointer walk-up -- NOT an O(V) mark-array scan. `depth` is
  // precomputed once per tree (see depth_of below) so this is O(1) extra
  // space per call, no per-call allocation of tree-sized scratch.
  while (depth[a] > depth[b]) a = parent[a];
  while (depth[b] > depth[a]) b = parent[b];
  while (a != b) { a = parent[a]; b = parent[b]; }
  return a;
}

inline int lca_of(const std::vector<int>& parent, const std::vector<int>& depth,
                   const std::vector<int>& nodes) {
  int cur = nodes[0];
  for (size_t i = 1; i < nodes.size(); ++i) cur = lca_pair(parent, depth, cur, nodes[i]);
  return cur;
}

inline std::vector<int> depth_of(const std::vector<int>& parent) {
  // depth[c] = 0 for a root, else depth[parent[c]]+1. Requires parents to
  // appear before children is NOT assumed; compute lazily with memoization
  // via path-halving (still O(V) total, once per tree, not once per query).
  int n = (int)parent.size();
  std::vector<int> d(n, -1);
  for (int c = 0; c < n; ++c) {
    if (d[c] >= 0) continue;
    std::vector<int> path;
    int x = c;
    while (x >= 0 && d[x] < 0) { path.push_back(x); x = parent[x]; }
    int base = (x >= 0) ? d[x] : -1;
    for (int i = (int)path.size() - 1; i >= 0; --i) d[path[i]] = ++base;
  }
  return d;
}

struct UnionFind {
  std::vector<int> p;
  explicit UnionFind(int n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
  void unite(int x, int y) { x = find(x); y = find(y); if (x != y) p[x] = y; }
};

}  // namespace localized_detail

inline LocalizedSelection select_atomic_actions_localized(
    const std::vector<int>& parent_in,
    const std::vector<double>& weight_in,
    const std::vector<AtomicAction>& actions,
    double lambda) {
  using namespace localized_detail;
  const int n_comm = (int)parent_in.size();  // REAL community count -- output sizing/guard threshold
  const int n_action = (int)actions.size();

  // Real DyLo hierarchies can be FORESTS (multiple parent==-1 roots -- e.g.
  // disconnected graph components each get their own top-level community),
  // which the earlier synthetic/counterexample tests never exercised (all
  // single-rooted). LCA-bucketing needs a single root to be well-defined:
  // an action whose two anchors sit in different real trees otherwise makes
  // lca_of() run off the end (parent[-1] is UB) -- confirmed as the actual
  // crash on real facebook data ("action anchor outside hierarchy", because
  // the resulting garbage LCA leaked through as an invalid anchor index).
  // Fix: add one virtual weight-0 super-root above every real root. Paying
  // for it is always free, so this cannot change the objective; it only
  // makes LCA total. Index n_comm (one past the real range) is reserved for
  // it and must never reach `out` (guarded at the single materialization
  // point below), since `out.*` is sized to the real n_comm.
  std::vector<int> parent(n_comm + 1);
  std::vector<double> weight(n_comm + 1);
  for (int c = 0; c < n_comm; ++c) {
    parent[c] = (parent_in[c] >= 0) ? parent_in[c] : n_comm;
    weight[c] = weight_in[c];
  }
  parent[n_comm] = -1;
  weight[n_comm] = 0.0;
  const std::vector<int> depth = localized_detail::depth_of(parent);  // O(V) ONCE per call, not per action

  // ---- bucket by LCA ----
  std::unordered_map<int, std::vector<int>> buckets;             // h -> action indices
  std::vector<int> h_of(n_action);
  std::vector<std::unordered_set<int>> branch_of(n_action);       // below-h, excl h
  for (int i = 0; i < n_action; ++i) {
    int h = lca_of(parent, depth, actions[i].anchors);
    h_of[i] = h;
    buckets[h].push_back(i);
    for (int s : actions[i].anchors) {
      int x = s;
      while (x != h && x >= 0) { branch_of[i].insert(x); x = parent[x]; }
    }
  }

  // ---- decomposability certificate: union-find over per-node ownership ----
  std::vector<int> bucket_ids;
  bucket_ids.reserve(buckets.size());
  for (auto& kv : buckets) bucket_ids.push_back(kv.first);
  std::unordered_map<int, int> comp_index;  // bucket lca -> index into bucket_ids
  for (size_t i = 0; i < bucket_ids.size(); ++i) comp_index[bucket_ids[i]] = (int)i;

  UnionFind uf((int)bucket_ids.size());
  std::unordered_map<int, std::vector<int>> owners_L;  // node -> bucket indices
  std::unordered_map<int, std::vector<int>> owners_A;  // node -> bucket indices (self+ancestors)
  for (int h : bucket_ids) {
    for (int a : ancestors_incl(parent, h))
      if (weight[a] > 0.0) owners_A[a].push_back(comp_index[h]);
  }
  for (auto& kv : buckets) {
    int h = kv.first;
    std::unordered_set<int> union_branch;
    for (int i : kv.second) union_branch.insert(branch_of[i].begin(), branch_of[i].end());
    for (int x : union_branch)
      if (weight[x] > 0.0) owners_L[x].push_back(comp_index[h]);
  }
  for (auto& kv : owners_L) {
    std::vector<int> all_owners = kv.second;
    auto ita = owners_A.find(kv.first);
    if (ita != owners_A.end()) all_owners.insert(all_owners.end(), ita->second.begin(), ita->second.end());
    if (all_owners.size() < 2) continue;
    for (size_t j = 1; j < all_owners.size(); ++j) uf.unite(all_owners[0], all_owners[j]);
  }

  std::unordered_map<int, std::vector<int>> components;  // root -> bucket indices
  for (size_t i = 0; i < bucket_ids.size(); ++i) components[uf.find((int)i)].push_back((int)i);

  LocalizedSelection out;
  out.selected_actions.assign(n_action, 0);
  out.dirty_nodes.assign(n_comm, 0);
  std::unordered_set<int> dirty_set;
  std::unordered_set<int> selected_set;

  // ---- singleton components: small local min-cut -> compressed super-action ----
  struct BucketValue { double net; std::vector<int> local_selected; };
  std::unordered_map<int, BucketValue> bucket_value;  // h -> value (only for singleton comps)
  std::vector<int> clean_h, coupled_h;

  for (auto& comp : components) {
    if (comp.second.size() > 1) {
      for (int bi : comp.second) coupled_h.push_back(bucket_ids[bi]);
      continue;
    }
    int h = bucket_ids[comp.second[0]];
    clean_h.push_back(h);
    const auto& idxs = buckets[h];
    std::unordered_set<int> branch_nodes_set;
    for (int i : idxs) branch_nodes_set.insert(branch_of[i].begin(), branch_of[i].end());
    if (branch_nodes_set.empty()) {
      double net = 0.0;
      std::vector<int> sel;
      for (int i : idxs) if (actions[i].gain > 0.0) { net += actions[i].gain; sel.push_back(i); }
      bucket_value[h] = {net, sel};
      continue;
    }
    std::vector<int> branch_nodes(branch_nodes_set.begin(), branch_nodes_set.end());
    std::sort(branch_nodes.begin(), branch_nodes.end());
    std::unordered_map<int, int> remap;
    for (size_t i = 0; i < branch_nodes.size(); ++i) remap[branch_nodes[i]] = (int)i;
    std::vector<int> local_parent(branch_nodes.size());
    std::vector<double> local_weight(branch_nodes.size());
    for (size_t i = 0; i < branch_nodes.size(); ++i) {
      int p = parent[branch_nodes[i]];
      local_parent[i] = (p >= 0 && remap.count(p)) ? remap[p] : -1;
      local_weight[i] = weight[branch_nodes[i]];
    }
    std::vector<AtomicAction> local_actions;
    for (int i : idxs) {
      AtomicAction a; a.gain = actions[i].gain;
      for (int x : actions[i].anchors) if (x != h) a.anchors.push_back(remap.at(x));
      local_actions.push_back(std::move(a));
    }
    ClosureSelection cs = select_atomic_actions(local_parent, local_weight, local_actions, lambda);
    out.network_nodes += (long)(local_parent.size() + 2);
    long e = 0;
    for (auto& a : local_actions) e += 1 + (long)a.anchors.size();
    for (size_t c = 0; c < local_parent.size(); ++c) { e += 1; if (local_parent[c] >= 0) e += 1; }
    out.network_edges += e;
    std::vector<int> sel;
    for (size_t i = 0; i < idxs.size(); ++i) if (cs.selected_actions[i]) sel.push_back(idxs[i]);
    bucket_value[h] = {cs.objective, sel};
  }

  // ---- final: combine clean super-actions + coupled buckets' real actions
  // in ONE reduced (pruned-to-touched-nodes) min-cut ----
  std::vector<AtomicAction> reduced;
  std::vector<std::pair<char, int>> origin;  // (0=clean h, 1=coupled action idx)
  for (int h : clean_h) {
    double net = bucket_value[h].net;
    if (net > 0.0) {
      AtomicAction a; a.gain = net; a.anchors = {h};
      reduced.push_back(std::move(a));
      origin.push_back({0, h});
    }
  }
  for (int h : coupled_h) {
    for (int i : buckets[h]) {
      reduced.push_back(actions[i]);
      origin.push_back({1, i});
    }
  }

  if (!reduced.empty()) {
    std::unordered_set<int> touched_set;
    for (auto& a : reduced)
      for (int x : a.anchors)
        for (int anc : ancestors_incl(parent, x)) touched_set.insert(anc);
    std::vector<int> touched(touched_set.begin(), touched_set.end());
    std::sort(touched.begin(), touched.end());
    std::unordered_map<int, int> remap;
    for (size_t i = 0; i < touched.size(); ++i) remap[touched[i]] = (int)i;
    std::vector<int> r_parent(touched.size());
    std::vector<double> r_weight(touched.size());
    for (size_t i = 0; i < touched.size(); ++i) {
      int p = parent[touched[i]];
      r_parent[i] = (p >= 0 && remap.count(p)) ? remap[p] : -1;
      r_weight[i] = weight[touched[i]];
    }
    std::vector<AtomicAction> r_actions;
    for (auto& a : reduced) {
      AtomicAction ra; ra.gain = a.gain;
      for (int x : a.anchors) ra.anchors.push_back(remap.at(x));
      r_actions.push_back(std::move(ra));
    }
    if (const char* e = std::getenv("DYLO_REDUCE_STAT"))
      fprintf(stderr, "[reduce] tree_nodes=%d actions_in=%d -> reduced_nodes=%zu reduced_actions=%zu"
                      " (clean_buckets=%zu coupled_buckets=%zu)\n",
              n_comm, (int)actions.size(), touched.size(), reduced.size(),
              clean_h.size(), coupled_h.size());
    ClosureSelection cs = select_atomic_actions(r_parent, r_weight, r_actions, lambda);
    out.network_nodes += (long)(r_parent.size() + 2);
    long e = 0;
    for (auto& a : r_actions) e += 1 + (long)a.anchors.size();
    for (size_t c = 0; c < r_parent.size(); ++c) { e += 1; if (r_parent[c] >= 0) e += 1; }
    out.network_edges += e;

    for (size_t j = 0; j < reduced.size(); ++j) {
      if (!cs.selected_actions[j]) continue;
      if (origin[j].first == 0) {
        for (int i : bucket_value[origin[j].second].local_selected) selected_set.insert(i);
        for (int a : ancestors_incl(parent, origin[j].second)) dirty_set.insert(a);
        for (int i : bucket_value[origin[j].second].local_selected)
          for (int x : branch_of[i]) dirty_set.insert(x);
      } else {
        selected_set.insert(origin[j].second);
      }
    }
    for (size_t c = 0; c < touched.size(); ++c)
      if (cs.dirty_nodes[c]) dirty_set.insert(touched[c]);
  }

  for (int i : selected_set) { out.selected_actions[i] = 1; out.selected_gain += actions[i].gain; }
  // dirty_set/touched may contain the virtual super-root (index n_comm,
  // weight 0) added above -- it isn't a real community, must never reach
  // out.dirty_nodes (sized to the real n_comm; writing index n_comm there
  // would be a one-past-the-end out-of-bounds write) or the cost sum
  // (harmless either way since its weight is 0, but excluded for clarity).
  for (int c : dirty_set) { if (c >= n_comm) continue; out.dirty_nodes[c] = 1; out.marginal_cost += weight[c]; }
  out.objective = out.selected_gain - lambda * out.marginal_cost;
  return out;
}


// ---------------------------------------------------------------------------
// Boundary-width probe for the two-sided joint-evaluation tree DP.
//
// That DP keeps, at every node u, a state over the leaves of subtree(u) whose
// partner anchor is OUTSIDE subtree(u) -- the "pending boundary" B_u. A leaf l
// is pending at u exactly when u lies strictly below LCA(a) on the path from l,
// for some action a containing l; anything whose anchors are all inside has
// already been settled at a lower LCA and need not be carried further up.
//
// The DP costs O(|T_A| * 2^b) with b = max_u |B_u|, i.e. it is EXPONENTIAL in
// the boundary width, whereas the min-cut it would replace is polynomial
// regardless. So b decides whether the DP is a speedup or a catastrophe, and b
// is a property of the workload, not of the algorithm. This function measures
// it on the real action sets so the decision is made on data.
struct BoundaryWidth {
  int max_width = 0;      // max_u |B_u|
  int max_children = 0;   // max_u |union of children's B_c| -- the join width
  long n_nodes = 0;       // nodes carrying a non-empty boundary
  double mean_width = 0.0;
};

inline BoundaryWidth measure_boundary_width(const std::vector<int>& parent_in,
                                            const std::vector<AtomicAction>& actions) {
  BoundaryWidth out;
  const int n = (int)parent_in.size();
  if (n == 0 || actions.empty()) return out;
  std::vector<int> parent(n + 1);
  for (int c = 0; c < n; ++c) parent[c] = (parent_in[c] >= 0) ? parent_in[c] : n;
  parent[n] = -1;
  const std::vector<int> depth = localized_detail::depth_of(parent);

  // B_u as a set of leaves, built by walking each anchor up to its action's LCA.
  std::unordered_map<int, std::unordered_set<int>> B;
  for (const auto& a : actions) {
    if (a.anchors.size() < 2) continue;          // single-anchor: settles at itself
    int h = localized_detail::lca_of(parent, depth, a.anchors);
    for (int s : a.anchors) {
      int x = s;
      while (x != h && x >= 0) { B[x].insert(s); x = parent[x]; }
    }
  }
  if (B.empty()) return out;
  long total = 0;
  for (auto& kv : B) {
    int w = (int)kv.second.size();
    out.max_width = std::max(out.max_width, w);
    total += w;
    out.n_nodes++;
  }
  // join width: the union a parent must reconcile across its children
  std::unordered_map<int, std::unordered_set<int>> R;
  for (auto& kv : B) {
    int p = parent[kv.first];
    if (p >= 0) R[p].insert(kv.second.begin(), kv.second.end());
  }
  for (auto& kv : R) out.max_children = std::max(out.max_children, (int)kv.second.size());
  out.mean_width = out.n_nodes ? (double)total / out.n_nodes : 0.0;
  return out;
}

}  // namespace dylo
