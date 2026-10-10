// Selector exactness check: OpenSet DP, min-cut, and enumeration on the same frozen instance.
//
// The paper's Theorem thm:dp claims that after the single-anchor reduction the maximization over
//     F(D) = sum_{c in D} ( g(c) - lambda * w(c) )
// ancestor-closed dirty sets is solved exactly by a linear-time tree DP. Previously only "DP and min-cut give the same
// committed set" had been checked, but both could be wrong -- a comparison with the **ground truth** was missing.
//
// This program enumerates **all** ancestor-closed subsets on small random forests (2^n upper bound, feasible for
// n<=20), takes the true optimum, and compares DP and min-cut case by case:
//   - whether the objective values are equal (1e-9 tolerance)
//   - whether the selected sets are identical (with tied optima, equal objectives pass and are counted separately)
//
// It also measures the time and peak memory of all three for the scalability comparison (sizes where DP/min-cut still run
// and enumeration is infeasible).
//
// Usage: selector_verify <mode> [args...]
//   exact  <max nodes> <cases> <seed>   enumeration comparison (nodes <= 20)
//   scale  <nodes> <cases> <seed>       DP vs min-cut only, time/memory
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dp.hpp"
#include "localized_closure.hpp"

using dylo::DPTree;
using dylo::open_set_dp;

namespace {

struct Forest {
  std::vector<int> parent;      // -1 = root
  std::vector<double> g, w;
};

// random forest: each node's parent is drawn from smaller indices, guaranteeing acyclicity and topological order.
Forest gen(int n, std::mt19937& rng, double gmax = 10.0, double wmax = 5.0) {
  Forest f;
  f.parent.resize(n); f.g.resize(n); f.w.resize(n);
  std::uniform_real_distribution<double> ug(-gmax, gmax), uw(0.0, wmax);
  for (int i = 0; i < n; i++) {
    // about 25% of the nodes are roots; the rest hang under a random predecessor
    f.parent[i] = (i == 0 || rng() % 4 == 0) ? -1 : (int)(rng() % i);
    f.g[i] = ug(rng);
    f.w[i] = uw(rng);
  }
  return f;
}

double eval(const Forest& f, const std::vector<char>& in, double lambda) {
  double s = 0.0;
  for (size_t i = 0; i < in.size(); i++)
    if (in[i]) s += f.g[i] - lambda * f.w[i];
  return s;
}

bool closed(const Forest& f, const std::vector<char>& in) {
  for (size_t i = 0; i < in.size(); i++)
    if (in[i] && f.parent[i] >= 0 && !in[f.parent[i]]) return false;
  return true;
}

// enumerate all 2^n subsets, keep only the ancestor-closed ones, and take the maximum F.
double brute(const Forest& f, double lambda, std::vector<char>& best) {
  int n = (int)f.parent.size();
  double bv = 0.0;                       // the empty set is always feasible, F=0
  best.assign(n, 0);
  for (long long m = 1; m < (1LL << n); m++) {
    std::vector<char> in(n, 0);
    for (int i = 0; i < n; i++) in[i] = (m >> i) & 1;
    if (!closed(f, in)) continue;
    double v = eval(f, in, lambda);
    if (v > bv + 1e-12) { bv = v; best = in; }
  }
  return bv;
}

std::vector<char> run_dp(const Forest& f, double lambda) {
  int n = (int)f.parent.size();
  DPTree T;
  T.children.assign(n, {});
  T.g = f.g; T.w = f.w;
  for (int i = 0; i < n; i++) {
    if (f.parent[i] >= 0) T.children[f.parent[i]].push_back(i);
    else T.roots.push_back(i);
  }
  return open_set_dp(T, lambda);
}

// generic maximum-weight closure (Picard 1976): node i has profit p_i = g_i - lambda*w_i,
// closure constraint: i selected => parent(i) selected. Build the graph and solve an s-t minimum cut; the source side is the optimal closure.
//   s -> i (cap = p_i)      if p_i > 0
//   i -> t (cap = -p_i)     if p_i < 0
//   i -> parent(i) (cap = inf)   enforces ancestor closure
// This is an **exact** method, independent of the DP, and serves as a second confirmation of the ground truth.
//
// Note: the harness's select_atomic_actions_localized cannot serve as the reference here --
// it buckets by LCA and solves decoupled buckets separately, which is correct in the driver (action anchors naturally
// cluster in a few subtrees), but if every node is passed as an independent single-anchor action, every node
// forms its own bucket, the buckets are deemed decoupled, the shared ancestor costs are split up, and the result is necessarily suboptimal.
namespace mf {
struct E { int to; double cap; int rev; };
struct Dinic {
  std::vector<std::vector<E>> g; std::vector<int> lv, it;
  explicit Dinic(int n) : g(n), lv(n), it(n) {}
  void add(int a, int b, double c) {
    g[a].push_back({b, c, (int)g[b].size()});
    g[b].push_back({a, 0.0, (int)g[a].size() - 1});
  }
  bool bfs(int s, int t) {
    std::fill(lv.begin(), lv.end(), -1);
    std::vector<int> q{s}; lv[s] = 0;
    for (size_t h = 0; h < q.size(); h++) {
      int v = q[h];
      for (auto& e : g[v])
        if (e.cap > 1e-12 && lv[e.to] < 0) { lv[e.to] = lv[v] + 1; q.push_back(e.to); }
    }
    return lv[t] >= 0;
  }
  double dfs(int v, int t, double f) {
    if (v == t) return f;
    for (int& i = it[v]; i < (int)g[v].size(); i++) {
      E& e = g[v][i];
      if (e.cap > 1e-12 && lv[v] < lv[e.to]) {
        double d = dfs(e.to, t, std::min(f, e.cap));
        if (d > 1e-12) { e.cap -= d; g[e.to][e.rev].cap += d; return d; }
      }
    }
    return 0.0;
  }
  void run(int s, int t) {
    while (bfs(s, t)) {
      std::fill(it.begin(), it.end(), 0);
      while (dfs(s, t, 1e18) > 1e-12) {}
    }
  }
};
}  // namespace mf

std::vector<char> run_cut(const Forest& f, double lambda) {
  int n = (int)f.parent.size(), S = n, T = n + 1;
  mf::Dinic d(n + 2);
  const double INF = 1e15;
  for (int i = 0; i < n; i++) {
    double p = f.g[i] - lambda * f.w[i];
    if (p > 0) d.add(S, i, p);
    else if (p < 0) d.add(i, T, -p);
    if (f.parent[i] >= 0) d.add(i, f.parent[i], INF);
  }
  d.run(S, T);
  // the set reachable from the source = the optimal closure
  std::vector<char> in(n, 0);
  std::vector<int> q{S}; std::vector<char> vis(n + 2, 0); vis[S] = 1;
  for (size_t h = 0; h < q.size(); h++)
    for (auto& e : d.g[q[h]])
      if (e.cap > 1e-12 && !vis[e.to]) { vis[e.to] = 1; q.push_back(e.to); }
  for (int i = 0; i < n; i++) in[i] = vis[i];
  return in;
}

long peak_kb() {
  FILE* fp = fopen("/proc/self/status", "r");
  if (!fp) return -1;
  char buf[256]; long v = -1;
  while (fgets(buf, sizeof buf, fp))
    if (!strncmp(buf, "VmHWM:", 6)) { sscanf(buf + 6, "%ld", &v); break; }
  fclose(fp);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s {exact|scale} <n> <cases> <seed>\n", argv[0]); return 2; }
  std::string mode = argv[1];
  int n = atoi(argv[2]), cases = atoi(argv[3]);
  std::mt19937 rng((uint32_t)atoi(argv[4]));
  const double LAM[] = {0.0, 0.1, 0.5, 1.0, 2.0};

  if (mode == "exact") {
    if (n > 22) { fprintf(stderr, "enumeration is limited to 22 nodes\n"); return 2; }
    long okv = 0, oks = 0, tie = 0, tot = 0, bad = 0;
    double tb = 0, td = 0, tm = 0;      // accumulated time of the three (ms)
    for (int c = 0; c < cases; c++) {
      int nn = 1 + (int)(rng() % n);
      Forest f = gen(nn, rng);
      for (double lam : LAM) {
        std::vector<char> bs;
        auto q0 = std::chrono::high_resolution_clock::now();
        double bv = brute(f, lam, bs);
        auto q1 = std::chrono::high_resolution_clock::now();
        auto d = run_dp(f, lam);
        auto q2 = std::chrono::high_resolution_clock::now();
        auto m = run_cut(f, lam);
        auto q3 = std::chrono::high_resolution_clock::now();
        tb += std::chrono::duration<double, std::milli>(q1 - q0).count();
        td += std::chrono::duration<double, std::milli>(q2 - q1).count();
        tm += std::chrono::duration<double, std::milli>(q3 - q2).count();
        double dv = eval(f, d, lam), mv = eval(f, m, lam);
        tot++;
        bool vok = std::fabs(dv - bv) < 1e-9 && std::fabs(mv - bv) < 1e-9;
        if (vok) okv++; else {
          bad++;
          if (bad <= 5)
            printf("  [mismatch] n=%d lam=%.2f  enumeration=%.9f DP=%.9f mincut=%.9f\n",
                   nn, lam, bv, dv, mv);
        }
        if (d == bs && m == bs) oks++;
        else if (vok) tie++;      // equal objective but different sets = tied optima
      }
    }
    printf("exact n<=%d: cases %ld  agree %ld (%.2f%%)  same set %ld  ties %ld  mismatches %ld  |  "
           "enumeration %.4f ms  DP %.4f ms  closure %.4f ms  |  enumeration/DP %.0fx  closure/DP %.1fx\n",
           n, tot, okv, 100.0 * okv / tot, oks, tie, bad,
           tb / tot, td / tot, tm / tot,
           tb / std::max(td, 1e-9), tm / std::max(td, 1e-9));
    return bad ? 1 : 0;
  }

  if (mode == "scale") {
    double tdp = 0, tcut = 0; long diff = 0, tot = 0;
    long m0 = peak_kb();
    for (int c = 0; c < cases; c++) {
      Forest f = gen(n, rng);
      for (double lam : LAM) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto d = run_dp(f, lam);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto m = run_cut(f, lam);
        auto t2 = std::chrono::high_resolution_clock::now();
        tdp += std::chrono::duration<double, std::milli>(t1 - t0).count();
        tcut += std::chrono::duration<double, std::milli>(t2 - t1).count();
        tot++;
        if (std::fabs(eval(f, d, lam) - eval(f, m, lam)) > 1e-9) diff++;
      }
    }
    printf("scale n=%d: cases %ld  DP %.4f ms/solve  min-cut %.4f ms/solve  ratio %.1fx  "
           "objective mismatches %ld  peak memory %ld KB (delta %ld)\n",
           n, tot, tdp / tot, tcut / tot, tcut / std::max(tdp, 1e-9), diff,
           peak_kb(), peak_kb() - m0);
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 2;
}
