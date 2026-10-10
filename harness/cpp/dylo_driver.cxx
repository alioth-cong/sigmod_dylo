// dylo_driver.cxx — per-batch wrapper around puzzlef's dynamic Louvain variants.
//
// Usage:
//   ./puzzlef_driver {df|ds} <initial.txt> <inc-base.txt> <numBatches> <outDir> <numThreads>
//
// inc-base.txt is logical; real files are <inc-base>_1.txt, <inc-base>_2.txt, ...
// initial.txt format: u\tv\tw\n (w ignored; treated as 1).
// inc_<k>.txt format: u\tv\tts\n (ts ignored).
//
// Output in <outDir>:
//   metrics.tsv  batch<TAB>latency_ms<TAB>modularity<TAB>num_communities
//   com_<k>.txt  one community per line, tab-separated node IDs (dynamo native format)
//
#include "inc/main.hxx"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <sys/stat.h>

using namespace std;

using K = uint32_t;
using V = TYPE;   // float

static string extend_filename(const string& path, const string& ext) {
  size_t i = path.rfind('.');
  return path.substr(0, i) + ext + "." + path.substr(i+1);
}

// Read a tab-separated edge list into the given DiGraph. Each line `u\tv[\t...]` is added as
// an undirected edge by inserting both u->v and v->u with weight 1.0.
template <class G>
static void readEdgeList(G& g, const string& path) {
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    K u, v;
    char *p = const_cast<char*>(line.c_str());
    char *e = nullptr;
    u = (K) strtoul(p, &e, 10); if (e == p) continue;
    p = e;
    while (*p && (*p == '\t' || *p == ' ')) ++p;
    v = (K) strtoul(p, &e, 10); if (e == p) continue;
    addEdgeOmpU(g, u, v, (V)1.0);
    addEdgeOmpU(g, v, u, (V)1.0);
  }
}

template <class G>
static vector<tuple<K, K, V>> readInsertionsList(const string& path) {
  vector<tuple<K, K, V>> ins;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    K u, v;
    char *p = const_cast<char*>(line.c_str());
    char *e = nullptr;
    u = (K) strtoul(p, &e, 10); if (e == p) continue;
    p = e;
    while (*p && (*p == '\t' || *p == ' ')) ++p;
    v = (K) strtoul(p, &e, 10); if (e == p) continue;
    ins.push_back({u, v, (V)1.0});
    ins.push_back({v, u, (V)1.0});
  }
  return ins;
}

template <class G, class A>
static double getMod(const G& y, const A& ans, double M) {
  auto fc = [&](auto u) { return ans.membership[u]; };
  return modularityBy(y, fc, M, 1.0);
}

// Write community structure in dynamo native format (one community per line).
template <class G>
static void writeCommunity(const G& y, const vector<K>& membership, const string& path) {
  // bucket by community id
  unordered_map<K, vector<K>> bag;
  size_t S = y.span();
  for (K u = 0; u < (K)S; ++u) {
    if (!y.hasVertex(u)) continue;
    K c = membership[u];
    bag[c].push_back(u);
  }
  ofstream g(path);
  for (auto& kv : bag) {
    auto& vec = kv.second;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i) g << '\t';
      g << vec[i];
    }
    g << '\n';
  }
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr, "Usage: %s {df|ds} <initial.txt> <inc-base.txt> <numBatches> <outDir> <numThreads>\n", argv[0]);
    return 2;
  }
  string algo = argv[1];
  string initialPath = argv[2];
  string incBase     = argv[3];
  int    numBatches  = atoi(argv[4]);
  string outDir      = argv[5];
  int    numThreads  = atoi(argv[6]);
  install_sigsegv();
  omp_set_num_threads(numThreads);
  fprintf(stderr, "[puzzlef:%s] threads=%d\n", algo.c_str(), numThreads);

  filesystem::create_directories(outDir);

  // 1) Load initial.
  DiGraph<K, None, V> x;
  readEdgeList(x, initialPath);
  updateOmpU(x);
  // ensure symmetric
  x = symmetrizeOmp(x);
  double M0 = edgeWeightOmp(x) / 2;
  fprintf(stderr, "[puzzlef] initial vertices=%zu edges=%zu\n",
          x.order(), x.size());

  // 2) Initial static Louvain.
  auto b0 = louvainStaticOmp(x, {5});
  auto BM = b0.membership;
  auto BV = b0.vertexWeight;
  auto BC = b0.communityWeight;

  ofstream metrics(outDir + "/metrics.tsv");
  metrics << fixed << setprecision(6);   // sub-ms (P2): b.time is ms as double
  metrics << "batch\tlatency_ms\tmodularity\tnum_communities\n";
  // Count distinct community ids
  unordered_map<K,int> hist;
  for (K c : BM) hist[c]++;
  metrics << "0\t" << (double)(b0.time) << '\t' << getMod(x, b0, M0) << '\t' << hist.size() << '\n';
  writeCommunity(x, BM, outDir + "/com_0.txt");
  metrics.flush();

  for (int k = 1; k <= numBatches; ++k) {
    string incPath = extend_filename(incBase, "_" + to_string(k));
    // Build y from x, apply insertions
    auto y = duplicate(x);
    vector<tuple<K, K, V>> deletions;  // empty (insertion-only)
    vector<tuple<K, K, V>> insertions = readInsertionsList<decltype(y)>(incPath);
    tidyBatchUpdateU(deletions, insertions, y);
    applyBatchUpdateOmpU(y, deletions, insertions);
    double M = edgeWeightOmp(y) / 2;

    auto finalize = [&](const auto& b) {
      copyValuesOmpW(BM, b.membership);
      copyValuesOmpW(BV, b.vertexWeight);
      copyValuesOmpW(BC, b.communityWeight);
      hist.clear();
      for (K c : BM) hist[c]++;
      metrics << k << '\t' << (double)(b.time) << '\t' << getMod(y, b, M) << '\t' << hist.size() << '\n';
      writeCommunity(y, BM, outDir + "/com_" + to_string(k) + ".txt");
      metrics.flush();
    };
    if (algo == "df") {
      auto b = louvainDynamicFrontierOmp(y, deletions, insertions, BM, BV, BC, {1});
      finalize(b);
    } else if (algo == "ds") {
      auto b = louvainDynamicDeltaScreeningOmp(y, deletions, insertions, BM, BV, BC, {1});
      finalize(b);
    } else {
      fprintf(stderr, "Unknown algo %s\n", algo.c_str());
      return 2;
    }

    // advance x = y
    x = move(y);
  }
  return 0;
}
