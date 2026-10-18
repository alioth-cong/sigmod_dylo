// Exact cost-aware selection for atomic, possibly multi-anchor actions.
//
// Each action a has a non-negative pending gain and a set of directly changed
// hierarchy nodes ("anchors").  Selecting a forces every anchor and all of its
// ancestors dirty.  A dirty hierarchy node c is paid once, even when several
// selected actions share it.  For fixed gains/footprints the objective is
//
//   max_S sum_{a in S} gain(a)
//           - lambda * sum_{c in union_{a in S} ancestors(anchors(a))} w(c).
//
// This is a maximum-weight closure problem.  The reduction below uses one
// positive node per action, one negative node per hierarchy cost, and infinite
// implication arcs action->anchor and child->parent.  A single s-t min-cut is
// exact for arbitrary action arity (one anchor, source+destination, split, ...).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dylo {

struct AtomicAction {
  double gain = 0.0;
  std::vector<int> anchors;
};

struct ClosureSelection {
  std::vector<char> selected_actions;
  std::vector<char> dirty_nodes;
  double selected_gain = 0.0;
  double marginal_cost = 0.0;  // unscaled sum of effective w(c)
  double objective = 0.0;      // selected_gain - lambda*marginal_cost
  double min_cut = 0.0;
};

namespace closure_detail {

struct FlowEdge {
  int to;
  int rev;
  double cap;
};

class Dinic {
 public:
  explicit Dinic(int n) : g_(n), level_(n), it_(n) {}

  void add_edge(int from, int to, double cap) {
    if (cap < -eps()) throw std::invalid_argument("negative flow capacity");
    cap = std::max(0.0, cap);
    FlowEdge f{to, static_cast<int>(g_[to].size()), cap};
    FlowEdge r{from, static_cast<int>(g_[from].size()), 0.0};
    g_[from].push_back(f);
    g_[to].push_back(r);
  }

  double max_flow(int s, int t) {
    double flow = 0.0;
    while (bfs(s, t)) {
      std::fill(it_.begin(), it_.end(), 0);
      while (true) {
        double f = dfs(s, t, std::numeric_limits<double>::infinity());
        if (f <= eps()) break;
        flow += f;
      }
    }
    return flow;
  }

  std::vector<char> source_side(int s) const {
    std::vector<char> seen(g_.size(), 0);
    std::queue<int> q;
    seen[s] = 1;
    q.push(s);
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (const FlowEdge& e : g_[u]) {
        if (e.cap > eps() && !seen[e.to]) {
          seen[e.to] = 1;
          q.push(e.to);
        }
      }
    }
    return seen;
  }

 private:
  static constexpr double eps() { return 1e-12; }

  bool bfs(int s, int t) {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<int> q;
    level_[s] = 0;
    q.push(s);
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (const FlowEdge& e : g_[u]) {
        if (e.cap > eps() && level_[e.to] < 0) {
          level_[e.to] = level_[u] + 1;
          q.push(e.to);
        }
      }
    }
    return level_[t] >= 0;
  }

  double dfs(int u, int t, double pushed) {
    if (u == t) return pushed;
    for (int& i = it_[u]; i < static_cast<int>(g_[u].size()); ++i) {
      FlowEdge& e = g_[u][i];
      if (e.cap <= eps() || level_[e.to] != level_[u] + 1) continue;
      double f = dfs(e.to, t, std::min(pushed, e.cap));
      if (f <= eps()) continue;
      e.cap -= f;
      g_[e.to][e.rev].cap += f;
      return f;
    }
    return 0.0;
  }

  std::vector<std::vector<FlowEdge>> g_;
  std::vector<int> level_;
  std::vector<int> it_;
};

}  // namespace closure_detail

// parent[c] is -1 at a root.  `prepaid[c]` means c is already dirty for a
// mandatory reason (e.g., content refresh), so selecting structural actions
// pays zero *marginal* cost for c.  The caller still adds the base prepaid cost
// when reporting end-to-end totals.
inline ClosureSelection select_atomic_actions(
    const std::vector<int>& parent,
    const std::vector<double>& weight,
    const std::vector<AtomicAction>& actions,
    double lambda,
    const std::vector<char>& prepaid = {}) {
  if (parent.size() != weight.size())
    throw std::invalid_argument("parent/weight size mismatch");
  if (lambda < 0.0) throw std::invalid_argument("lambda must be non-negative");
  if (!prepaid.empty() && prepaid.size() != parent.size())
    throw std::invalid_argument("prepaid/weight size mismatch");

  const int n_comm = static_cast<int>(parent.size());
  const int n_action = static_cast<int>(actions.size());
  const int action_base = 0;
  const int comm_base = n_action;
  const int source = n_action + n_comm;
  const int sink = source + 1;

  double total_gain = 0.0;
  double total_penalty = 0.0;
  for (const AtomicAction& a : actions) {
    if (a.gain < -1e-12) throw std::invalid_argument("negative action gain");
    total_gain += std::max(0.0, a.gain);
    for (int c : a.anchors)
      if (c < 0 || c >= n_comm)
        throw std::out_of_range("action anchor outside hierarchy");
  }
  for (int c = 0; c < n_comm; ++c) {
    if (weight[c] < -1e-12) throw std::invalid_argument("negative node weight");
    if (parent[c] < -1 || parent[c] >= n_comm || parent[c] == c)
      throw std::out_of_range("invalid hierarchy parent");
    if (prepaid.empty() || !prepaid[c])
      total_penalty += lambda * std::max(0.0, weight[c]);
  }

  // Strictly larger than every finite cut.  This makes every implication hard
  // without using an arbitrary machine-scale sentinel.
  const double inf = total_gain + total_penalty + 1.0;
  closure_detail::Dinic flow(sink + 1);

  for (int i = 0; i < n_action; ++i) {
    flow.add_edge(source, action_base + i, std::max(0.0, actions[i].gain));
    for (int c : actions[i].anchors)
      flow.add_edge(action_base + i, comm_base + c, inf);
  }
  for (int c = 0; c < n_comm; ++c) {
    const double effective_w =
        (!prepaid.empty() && prepaid[c]) ? 0.0 : std::max(0.0, weight[c]);
    flow.add_edge(comm_base + c, sink, lambda * effective_w);
    if (parent[c] >= 0)
      flow.add_edge(comm_base + c, comm_base + parent[c], inf);
  }

  ClosureSelection out;
  out.min_cut = flow.max_flow(source, sink);
  const std::vector<char> reachable = flow.source_side(source);
  out.selected_actions.assign(n_action, 0);
  out.dirty_nodes.assign(n_comm, 0);

  for (int i = 0; i < n_action; ++i) {
    if (reachable[action_base + i]) {
      out.selected_actions[i] = 1;
      out.selected_gain += actions[i].gain;
    }
  }
  for (int c = 0; c < n_comm; ++c) {
    if (reachable[comm_base + c]) {
      out.dirty_nodes[c] = 1;
      if (prepaid.empty() || !prepaid[c]) out.marginal_cost += weight[c];
    }
  }
  out.objective = out.selected_gain - lambda * out.marginal_cost;
  return out;
}

}  // namespace dylo
