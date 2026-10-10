// DyLo cost-aware incremental driver (Algorithm 1) over a batch stream.
//   ./dylo_driver {louvain|slm|leiden} <initial.txt> <inc-base> <numBatches> <outDir>
// Streams inc_<k>.txt; maintains the partition + hierarchy incrementally, confining
// re-optimization to the affected region and using the open-set DP (Alg 2) to decide
// which communities to re-summarize. Emits metrics.tsv (batch latency_ms modularity
// num_communities), com_<k>.txt (flat), com_<k>_L*.txt (hierarchy), and dirty.tsv
// (batch dirty_cost) for the re-summarization-cost metric.
//
// Two modes:
//   default (fast)        — leaf-only affected-region update, flat output. Used for the
//                           latency/quality tables. lambda has NO effect here.
//   DYLO_EMIT_HIER=1       — cost-aware: maintain the finest partition incrementally, rebuild
//                           the summary hierarchy, run exact atomic multi-anchor selection
//                           with DYLO_LAMBDA to COMMIT/WITHHOLD moves, and emit
//                           com_<k>_L*.txt so the harness computes hierarchical re-summ cost.
//                           lambda=0 -> commit all (max Q/cost); lambda=inf -> freeze (drift).
// EXPERIMENTAL BRANCH -- ports the Margin-Certificate Lazy Maintenance
// scheme (validated for correctness in driver_margin_certificate.cxx,
// FAST mode: 0 missed detections across 6 datasets/~350 batches after
// fixing 4 real gaps) into the HIER/cost-aware path, replacing L1's simple
// per-vertex marginH/pertH scalar gate with the per-action certificate
// system. Everything else (multi-round loop, make_cost_tree, the exact
// min-cut selector, incremental DeltaQ, L3) is UNCHANGED from
// driver_lazy_incq.cxx. Requested specifically so the comparison is
// against the cost-aware baseline at matched batch sizes (DYLO_EMIT_HIER=1
// on both sides), not the unrelated FAST-mode margin gate.
#include "detector_fixed.hpp"
#include "dp.hpp"
#include "closure.hpp"
#include "localized_closure.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <unordered_set>
#include <set>
#include <queue>
#include <tuple>
using namespace dylo;
static const double GAMMA_H=1.0;

static std::string extend(const std::string& p, const std::string& e){ size_t i=p.rfind('.'); return p.substr(0,i)+e+"."+p.substr(i+1); }

// Cost tree of the last committed hierarchy. A move between two existing
// communities anchors both the source and the destination. Selecting an action
// therefore pays for the union of both ancestor paths exactly once.
struct HierCostTree {
  std::vector<int> parent;
  std::vector<double> weight;
  std::vector<int> vertexLeaf;
  std::vector<std::unordered_map<int,int>> ids;   // ids[level][comm] -> node id
  // Reverse of `ids`. Node ids are only valid for ONE build of the tree, but a
  // batch's re-summarization set has to survive the rebuilds that happen
  // between rounds, so the set is accumulated as (level, community) -- which is
  // stable for as long as the partition is not renumbered -- and priced against
  // whichever tree is current when the batch ends.
  std::vector<std::pair<int,int>> nodeKey;        // node id -> (level, comm)
};

// Fixed per-summary overhead alpha (DYLO_ALPHA, default 0 = original model).
// The original cost model w(c) = #members (leaf) / #children (internal)
// charges a community purely by how much CONTENT the LLM must read to
// regenerate its summary. On the real NarrativeQA index that proxy is
// faithful for input tokens (tokens ~= 16.97*w + 3.8), and a singleton
// summary genuinely is ~5 tokens -- so the model is NOT under-charging in
// token terms. What it misses is that a newly-arrived vertex's sentinel
// community is not part of committedHierarchy at all, so make_cost_tree
// never sees it and "park the arrival in its own trivial community" costs
// exactly zero, while "merge it into D" pays D's full ancestor closure.
// That asymmetry lets a large lambda drive Cost toward zero by parking
// every arrival (up to 77.7% of communities become singletons on Epinions)
// while modularity barely notices (Q moves 0.03%). Exp-9 separately
// measured that per-LLM-call latency is dominated by fixed request
// overhead rather than input length, i.e. a real per-call cost the
// input-token proxy misses. alpha prices that per-call overhead; see the
// action-build site for where it is applied and why it cannot be applied
// to tree-node weights here.
static double dylo_alpha(){
  static const double A = [](){ const char* e=std::getenv("DYLO_ALPHA"); return e?std::atof(e):0.0; }();
  return A;
}

static HierCostTree make_cost_tree(const Graph& g, const Hierarchy& H){
  HierCostTree out;
  if(H.levels.empty()) return out;
  const int nlev=(int)H.levels.size();

  out.ids.assign(nlev,{});
  auto& ids = out.ids;
  auto get_id=[&](int level,int comm){
    auto it=ids[level].find(comm);
    if(it!=ids[level].end()) return it->second;
    int id=(int)out.parent.size();
    ids[level][comm]=id;
    out.parent.push_back(-1);
    out.weight.push_back(0.0);       // alpha is NOT applied here -- see the note at the action-build site
    out.nodeKey.push_back({level,comm});
    return id;
  };
  out.vertexLeaf.assign(g.n,-1);
  const int nv=std::min(g.n,(int)H.levels[0].size());
  for(int v=0;v<nv;v++){
    if(g.k[v]<=0) continue;
    int id=get_id(0,H.levels[0][v]);
    out.vertexLeaf[v]=id;
    out.weight[id]+=1.0;                         // leaf cost = number of vertices
  }
  for(int li=0;li+1<nlev;li++){
    std::unordered_map<int,int> par;
    for(int v=0;v<nv;v++){
      if(g.k[v]<=0) continue;
      int child=H.levels[li][v];
      if(!par.count(child)) par[child]=H.levels[li+1][v];
    }
    for(const auto& [child_comm,parent_comm]:par){
      int child=get_id(li,child_comm), parent=get_id(li+1,parent_comm);
      out.parent[child]=parent;
      out.weight[parent]+=1.0;                   // internal cost = number of children
    }
  }
  return out;
}

// Edge reader. NOTE: the streaming format is u<TAB>v<TAB><batch_idx> (prep_data.py);
// the 3rd column is the BATCH INDEX, not an edge weight. The dynamic graph is
// unweighted, so we IGNORE column 3 and use weight 1 (matching the static references
// leiden-static / slm-static, which treat the stream as unweighted). Reading column 3
// as a weight inflates m and modularity (batch-k edges would get weight k).
template<class F> static void read_edges(const std::string& path, F&& cb){
  // Fast manual integer parsing (istringstream-per-line was ~1ms/batch of pure overhead).
  std::ifstream f(path, std::ios::binary);
  if(!f) return;
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const char* p = buf.data(); const char* end = p + buf.size();
  while(p<end){
    while(p<end && (*p=='\n'||*p=='\r'||*p==' '||*p=='\t')) ++p;
    if(p>=end) break;
    if(*p=='#'){ while(p<end && *p!='\n') ++p; continue; }
    long u=0; bool any=false; while(p<end && *p>='0' && *p<='9'){ u=u*10+(*p-'0'); ++p; any=true; }
    if(!any){ while(p<end && *p!='\n') ++p; continue; }
    while(p<end && (*p==' '||*p=='\t')) ++p;
    long v=0; while(p<end && *p>='0' && *p<='9'){ v=v*10+(*p-'0'); ++p; }
    while(p<end && *p!='\n') ++p;                 // skip rest of line (3rd col = batch idx, ignored)
    cb((K)u,(K)v,1.0);
  }
}

int main(int argc,char**argv){
  if(argc<6){ fprintf(stderr,"usage: %s {louvain|slm|leiden} initial inc-base nBatches outDir [threads_ignored]\n  lambda/K/R from env: DYLO_LAMBDA (default 0), DYLO_AUDIT_K (8), DYLO_REBUILD_R (1e9)\n  DYLO_EMIT_HIER=1 -> cost-aware hierarchical mode (lambda active, emits com_k_L*.txt)\n",argv[0]); return 2; }
  std::string algo=argv[1], initial=argv[2], incBase=argv[3]; int nB=atoi(argv[4]);
  std::string outDir=argv[5];                                    // argv[6] threads = ignored (harness convention)
  const char* lambdaEnv=std::getenv("DYLO_LAMBDA");
  const char* auditEnv=std::getenv("DYLO_AUDIT_K");
  const char* rebuildEnv=std::getenv("DYLO_REBUILD_R");
  const char* hierEnv=std::getenv("DYLO_EMIT_HIER");
  const char* noComEnv=std::getenv("DYLO_NO_COM");
  double lambda = std::atof(lambdaEnv?lambdaEnv:"0");
  int auditK    = std::atoi(auditEnv?auditEnv:"8");
  double R      = std::atof(rebuildEnv?rebuildEnv:"1e18");  // default: never rebuild (audit off)
  const char* sigmaEnv=std::getenv("DYLO_SIGMA");
  const double SIGMA = sigmaEnv?std::atof(sigmaEnv):1.0;    // default 1.0 = structural trigger off
  const bool RECLAIM = std::getenv("DYLO_RECLAIM")!=nullptr;
  // SIGMA-ONLY audit. The audit currently does two unrelated jobs: it measures
  // drift (L2) and it measures the singleton fraction (the structural trigger).
  // Only the first needs the expensive part -- one whole-graph local_moving
  // pass plus two whole-graph modularity() calls, profiled at 65-83% of audit
  // time. The singleton fraction is an O(#communities) count over `leaf`.
  // With DYLO_SIGMA_ONLY the audit computes sfrac and runs targeted
  // reclamation, and skips the probe, the drift accumulator and the
  // drift-triggered rebuild entirely.
  const bool SIGMA_ONLY = std::getenv("DYLO_SIGMA_ONLY")!=nullptr;
  const bool LAZY_CERT = [](){ const char* e=std::getenv("DYLO_LAZY_CERT"); return e?std::atoi(e)!=0:true; }();
  // Patch the summary cost tree in place after a committed round instead of
  // rebuilding it. A round moves a handful of vertices between EXISTING
  // communities, so the tree's shape is unchanged except that two leaves gain
  // and lose a member and, if a leaf empties, it and possibly its ancestors
  // leave the tree. Rebuilding was O(N * levels) per accepted round and is a
  // large part of the round phase on big graphs.
  const bool LAZY_RECLAIM = [](){ const char* e=std::getenv("DYLO_LAZY_RECLAIM"); return e?std::atoi(e)!=0:true; }();
  // Use the tree DP instead of the min-cut solver when actions are single-anchored.
  // Charge the deferred source's amortised re-summarization to the action.
  const bool SRC_CHARGE = [](){ const char* e=std::getenv("DYLO_SRC_CHARGE"); return e?std::atoi(e)!=0:true; }();
  const bool USE_TREE_DP = [](){ const char* e=std::getenv("DYLO_TREE_DP"); return e?std::atoi(e)!=0:true; }();
  // R2 ablation arm (DYLO_GREEDY=1): replace the OpenSet DP with per-destination greedy.
  // OpenSet (dp.hpp, Alg 2) gives the **exact** optimum over ancestor-closed dirty sets; it does two things
  // greedy cannot:
  //   (a) the cost of a shared ancestor is amortized across siblings -- k destinations under one parent pay w(parent) once;
  //   (b) it can accept a parent with g-lambda*w < 0 as long as the surplus of its subtree pays it back
  //       (exactly the sum max(open(child),0) term in open(c)).
  // greedy decides each destination independently, each paying its whole ancestor chain, so (a)(b) are lost. It is
  // the honest floor of "what if there were no Alg 2", not another speed switch.
  const bool GREEDY = [](){ const char* e=std::getenv("DYLO_GREEDY"); return e && std::atoi(e)!=0; }();
  const bool INC_TREE = [](){ const char* e=std::getenv("DYLO_INC_TREE"); return e?std::atoi(e)!=0:true; }();
  const bool VALIDATE_TREE = std::getenv("DYLO_VALIDATE_TREE")!=nullptr;
  const bool VALIDATE_QC   = std::getenv("DYLO_VALIDATE_QC")!=nullptr;
  // ---------------------------------------------------------------------
  // (1) SIZE PREFILTER (DYLO_PRUNE, default on).
  // The selector's job is to maximize F = DeltaQ - lambda*Cost, but every
  // candidate move still had to be built, pushed, propagated to neighbours
  // and handed to the min-cut solver before being rejected. A move into a
  // community of size |D| pays at least a re-summarization proportional to
  // |D|, so a move whose local modularity gain cannot even cover
  // lambda*|D| can never be part of an accepted set. Testing that at the
  // point the move is created costs one hash lookup and removes the
  // candidate before it triggers the neighbour-enqueue cascade.
  //
  // NOTE ON ADMISSIBILITY: |D| is the destination COMMUNITY's member count,
  // while the action's true anchor is the destination FINE LEAF, whose
  // ancestor closure is typically much smaller than |D| (on Facebook, ~22
  // vs ~224). The filter is therefore STRICTER than the model it
  // approximates -- it can discard a move the full selector would have
  // accepted. That is a deliberate speed/quality trade; DYLO_PRUNE=0
  // restores the exact behaviour, and the Q columns of the ablation are
  // what decide whether the trade is worth taking.
  const bool PRUNE = [](){ const char* e=std::getenv("DYLO_PRUNE"); return e?std::atoi(e)!=0:true; }();
  // Which size stands in for the destination's re-summarization cost.
  //   "comm" -- the destination COMMUNITY's member count. Simple, but under the
  //            top-down tree the community sits at levels.back() while the
  //            action's anchor is a fine leaf at levels[0], so this OVERSTATES
  //            the cost (Facebook: 224 vs ~22) and the filter is stricter than
  //            the model it stands for.
  //   "leaf" -- the destination FINE LEAF's member count, i.e. the weight of
  //            the node the action actually anchors. closure(D) is that weight
  //            plus its ancestors, so this is a true LOWER bound: the filter can
  //            no longer discard a move the full selector would have accepted.
  // Aggregating the tree at a lower resolution would unify the two by putting
  // the maintained partition at levels[0], but measured 10.7x (Facebook) and
  // 14.5x (Cit-HepTh) higher re-summarization cost, because the leaves become
  // whole communities. Reading the fine leaf directly gets the same
  // admissibility for nothing.
  const bool PRUNE_LEAF = [](){ const char* e=std::getenv("DYLO_PRUNE_MODE");
                                return e ? std::string(e)=="leaf" : true; }();
  // ---------------------------------------------------------------------
  // (2) TRANSIENT OVERLAP / DEFERRED SOURCE REMOVAL (DYLO_DEFER, default on).
  // A move v: A -> B currently anchors BOTH communities, because v's
  // departure changes A's member set as much as its arrival changes B's.
  // Two anchors per action is exactly what makes the action set couple
  // across the tree and blocks the per-node DP.
  //
  // Under transient overlap the index does not remove v from A right away.
  // B's summary must be regenerated (v arrived, it must be described);
  // A's summary is left listing v. It is not WRONG -- v is still described
  // there -- only redundant. A is refreshed later, in one batch, once its
  // residual fraction reaches THETA. Each action therefore has a single
  // anchor and the tree DP applies directly.
  //
  // The partition is untouched by this: Q, community counts and the
  // hierarchy are identical by construction. Only the refresh SCHEDULE
  // changes, which is why the deferral's effect is invisible to a scorer
  // that reads partitions alone and has to be measured from the schedule
  // the driver actually emits (resum_*.tsv).
  const bool DEFER = [](){ const char* e=std::getenv("DYLO_DEFER"); return e?std::atoi(e)!=0:true; }();
  // ---------------------------------------------------------------------
  // (3) RESOLUTION-AGGREGATED HIERARCHY (DYLO_HIER_GAMMA, default 1.0 = off).
  // The top-down tree puts the MAINTAINED partition at levels.back() and the
  // recursively split fine communities at levels[0], so an action's anchor
  // (a fine leaf) and the community the size prefilter measures (|D| at the
  // maintained level) live at opposite ends of the tree -- which is why the
  // prefilter is stricter than the model it approximates.
  //
  // Aggregating at a LOWER resolution inverts that: the maintained partition
  // becomes levels[0], i.e. the summary-tree leaves, and coarser levels are
  // built above it. Then |D| IS the anchor leaf's weight, so the prefilter
  // becomes a true lower bound on closure(D) rather than an over-estimate.
  //
  // Aggregation at gamma=1 cannot do this -- the partition is already a
  // modularity optimum -- but a lower gamma prefers larger communities and
  // merges them: measured on Facebook, 1 level at gamma=1 becomes 3 at
  // gamma=0.1. Only communities sharing an edge can merge, so the depth
  // still bottoms out at the community graph's connected components.
  const double HIER_GAMMA = [](){ const char* e=std::getenv("DYLO_HIER_GAMMA"); return e?std::atof(e):1.0; }();
  // (4) COMBINED TREE (DYLO_COMBINED=1). Split downward AND aggregate upward
  // from the detected communities, then maintain levels[0] -- the fine split
  // leaves. Gets the aggregated mode's unified anchor/prefilter definition
  // without its expensive leaves. Uses the TOPDOWN=0 convention, since the
  // maintained partition is levels[0]; DYLO_TD_MAXSIZE and DYLO_HIER_GAMMA
  // control the two halves.
  const bool COMBINED = [](){ const char* e=std::getenv("DYLO_COMBINED"); return e && std::atoi(e)!=0; }();
  const double THETA = [](){ const char* e=std::getenv("DYLO_THETA"); return e?std::atof(e):0.2; }();
  // What decides that a deferred community has drifted far enough to be worth
  // regenerating.
  //   "frac"    -- residual members / listed members >= THETA. Simple, but it
  //                measures churn rather than consequence: a community that
  //                loses a peripheral member is charged the same as one whose
  //                structure changed.
  //   "quality" -- the community's own modularity term
  //                   q_A = in_A/2m - (sigma_A/2m)^2
  //                has moved by more than DYLO_DEFER_EPS, relative to its value
  //                when the summary was last written. A summary describes a
  //                community's structure, so it goes stale when that structure
  //                changes, not when its roster does.
  const bool DEFER_QUALITY = [](){ const char* e=std::getenv("DYLO_DEFER_MODE");
                                   return e ? std::string(e)=="quality" : true; }();
  const double DEFER_EPS = [](){ const char* e=std::getenv("DYLO_DEFER_EPS"); return e?std::atof(e):0.05; }();
  // INCREMENTAL RECLAMATION. Profiled 2026-08-06: of the time targeted
  // reclamation costs, moving the vertices is 0.0-0.1% and finding them is
  // 0.3-0.4%. The other 99%+ is the clean-up afterwards -- rebuilding the whole
  // hierarchy (50-57%) and rescanning every adjacency to rebuild kvComm
  // (34-37%). Both exist only because build_levels_from_leaf RENUMBERS
  // communities, which invalidates every community-keyed structure.
  //
  // But reclamation never creates or destroys a community: it moves parked
  // singletons into EXISTING neighbours. Keeping the ids stable makes the
  // clean-up proportional to the vertices actually moved. The certificate
  // rebuild is deliberately left full: it is only 8-12% and re-deriving it from
  // the updated kvComm keeps the "0 missed detections" guarantee without
  // needing a separate invalidation argument.
  const bool FAST_RECLAIM = std::getenv("DYLO_FAST_RECLAIM")!=nullptr;
  // TOPDOWN: build the summary tree by recursive splitting rather than by
  // aggregation. See detector.hpp::build_levels_topdown for why aggregation
  // yields exactly one level on these graphs. With it on, levels[0] are fine
  // split communities (the summary-tree LEAVES) and levels.back() is the
  // partition DyLo maintains, so a move's two anchors are the vertex's old and
  // new FINE leaf and their ancestor closures automatically cover the
  // maintained-level communities they sit under.
  // Value-aware, unlike the presence-only test the other drivers use: the
  // resolution-aggregated mode has to be selectable by setting this to 0,
  // and a presence-only test silently ignores that -- it made an entire
  // gamma sweep re-run the top-down configuration five times.
  const bool TOPDOWN = [](){ const char* e=std::getenv("DYLO_TOPDOWN"); return e && std::atoi(e)!=0; }();
  // FINAL AUDIT. The structural repair only runs at audit points, so whatever
  // accumulates after the LAST one is what the index is left holding. Measured
  // on Twitter (135 batches, K=8): the singleton fraction is 0.0% through batch
  // 128 -- every audit clears it -- then climbs to 26.1% over the remaining 7
  // unaudited batches, and the stream ends there. The reported figure is
  // therefore a sample of the audit phase (nB mod K), not a steady state, which
  // is why sweeping K moves it around non-monotonically instead of reducing it.
  // Forcing one audit on the final batch makes the end state phase-independent;
  // in deployment this is "clean the index before serving queries", and after
  // incremental reclamation it costs a few milliseconds.
  const bool FINAL_AUDIT = std::getenv("DYLO_FINAL_AUDIT")!=nullptr;
  const int TD_MAXSZ = [](){ const char* e=std::getenv("DYLO_TD_MAXSIZE"); return e?std::atoi(e):25; }();
  const bool VALIDATE_RECLAIM = std::getenv("DYLO_VALIDATE_RECLAIM")!=nullptr;
  const bool VALIDATE_AUDIT = std::getenv("DYLO_VALIDATE_AUDIT")!=nullptr;
  long statRepMiss=0, statRepStale=0, statSinglesMismatch=0, statSinglesOK=0, statRepDiff=0, statRepSame=0;  // targeted singleton reclamation instead of full rebuild
  const char* peelEnv=std::getenv("DYLO_PEEL");
  const bool PEEL = !peelEnv || std::atoi(peelEnv)!=0;   // local-optimality repair, on by default
  bool emitHier = std::atoi(hierEnv?hierEnv:"0") != 0;
  bool noCom    = std::atoi(noComEnv?noComEnv:"0") != 0;  // skip com_*.txt I/O (latency-only runs)
  bool lamInf   = lambdaEnv && std::string(lambdaEnv)=="inf";
  bool legacySelector = std::getenv("DYLO_SELECTOR") &&
                        std::string(std::getenv("DYLO_SELECTOR"))=="legacy";
  // "<inst>-fs" => fair from-scratch baseline: re-run DyLo's OWN C++ detector per batch,
  // no incremental state (same detector as DyLo, so the quality/speedup comparison is
  // same-implementation, not C++ vs Java). Requested 2026-05-29 (fair from-scratch).
  bool fromScratch = algo.size()>3 && algo.substr(algo.size()-3)=="-fs";
  if(fromScratch) algo = algo.substr(0, algo.size()-3);
  Instance inst = algo=="louvain"?Instance::LOUVAIN : algo=="slm"?Instance::SLM : Instance::LEIDEN;
  std::filesystem::create_directories(outDir);

  HierCostTree costTreeCache; bool costTreeValid=false;   // invalidated whenever committedHierarchy changes

  // (1) member count per community of the maintained partition, for the size
  // prefilter. Maintained incrementally on commit; rebuilt only where `leaf`
  // is reassigned wholesale (audit rebuild / reclamation renumber).
  std::unordered_map<int,int> commSize;
  // Which vertices commSize has already counted. A vertex that existed but was
  // isolated is skipped at initialisation (k=0) and is not in the arrival
  // range either, so gaining its first edge would leave its community
  // uncounted -- measured on Facebook as 20 live communities against the true
  // 29, which hid 9 of the 11 singletons and stopped reclamation firing.
  std::vector<char> inCommSize;
  std::unordered_map<int,int> fineSize;           // members per FINE community, for the staleness fraction
  // (2) residual entities per FINE community: members that have left but whose
  // summary still lists them. Keyed by fine-level community id rather than by
  // cost-tree node id, because node ids do not survive a tree rebuild. A full
  // rebuild regenerates every summary, so it legitimately clears all residuals.
  std::unordered_map<int,int> staleCnt;
  std::unordered_map<int,int> staleParent;        // fine community -> its level-1 label, captured before it can vanish
  // Internal edge weight per community, in modularity()'s convention (each
  // intra-community edge counted twice). Together with ctotH this makes a
  // community's own modularity term
  //     q_c = in_c/2m - (sigma_c/2m)^2
  // an O(1) lookup. Evaluating it by scanning members and their adjacency, as
  // the first version did, cost a per-batch pass over the vertices and made the
  // quality-based refresh rule 10x slower than the partition it was judging.
  std::unordered_map<int,double> cinH;
  std::unordered_map<int,double> qAtRefresh;
  // Vertices the cost tree does not cover yet. They accumulate across batches
  // -- a batch with no accepted round runs no patch, so its arrivals stay
  // pending -- and are drained by the next patch or cleared by a full rebuild.
  std::vector<int> pendingTreeV;      // fine community -> its modularity term when last summarized
  long statFlushVanished=0, statFlushQuality=0;
  long statDeferred=0, statFlushed=0, statFlushEmpty=0; double statFlushLive=0.0;             // moves whose source was deferred / theta-flushes
  long statPruned=0;
  // Affected-region volume in the cost-aware path. The claim that the region
  // stays a small fraction of |E| was previously measured on the maintenance
  // path only; with the margin gate active here it has to be measured here too.
  double statVolA=0.0; long statRegionBatches=0;                              // candidate moves killed by the size prefilter
  // initial graph + from-scratch detect
  Graph g; read_edges(initial,[&](uint32_t u,uint32_t v,double w){ g.add_edge(u,v,w); });
  fprintf(stderr,"[dylo:%s] initial %d V, m=%.0f, lambda=%s, hier=%d\n",
          algo.c_str(),g.n,g.m2/2, lamInf?"inf":std::to_string(lambda).c_str(), (int)emitHier);
  std::mt19937 rng(123);

  auto count_comm=[&](const std::vector<int>& memb){
    std::unordered_map<int,int> s; for(int i=0;i<(int)memb.size();i++) if(g.k[i]>0) s[memb[i]]; return (int)s.size(); };
  auto write_com=[&](int k,const std::vector<int>& memb){
    std::unordered_map<int,std::vector<int>> bag; for(int i=0;i<(int)memb.size();i++) if(g.k[i]>0) bag[memb[i]].push_back(i);
    std::ofstream o(outDir+"/com_"+std::to_string(k)+".txt");
    for(auto&[c,vs]:bag){ for(size_t i=0;i<vs.size();i++){ if(i)o<<'\t'; o<<vs[i]; } o<<'\n'; }
  };
  // Emit a full dendrogram (levels[0]=L0 finest ... back=top) as com_<k>_L*.txt.
  auto write_levels=[&](int k,const Hierarchy& H){
    for(int li=0; li<(int)H.levels.size(); li++){
      const auto& memb = H.levels[li];
      std::unordered_map<int,std::vector<int>> bag;
      for(int i=0;i<(int)memb.size() && i<g.n;i++) if(g.k[i]>0) bag[memb[i]].push_back(i);
      std::ofstream o(outDir+"/com_"+std::to_string(k)+"_L"+std::to_string(li)+".txt");
      for(auto&[c,vs]:bag){ for(size_t i=0;i<vs.size();i++){ if(i)o<<'\t'; o<<vs[i]; } o<<'\n'; }
    }
  };

  std::ofstream metrics(outDir+"/metrics.tsv"); metrics<<std::fixed<<std::setprecision(6);
  metrics<<"batch\tlatency_ms\tmodularity\tnum_communities\trounds\n";
  std::ofstream resum(outDir+"/resum.tsv"); resum<<std::fixed<<std::setprecision(6);
  resum<<"batch\tall_dirty\tn_nodes\tcost_abs\ttree_weight\tcost_norm\n";
  // Batches in which a full rebuild fired. A rebuild renumbers/reshuffles
  // the whole partition, so that batch's dirty set is essentially the whole
  // tree and its re-summarization cost (~1.0) swamps the per-batch average.
  // Those tokens are genuinely spent, so this file does NOT make the cost
  // disappear -- it lets the two be reported SEPARATELY: incremental-only
  // cost (comparable to GreMod/DynaMo, which have no rebuild mechanism at
  // all) and periodic-rebuild cost (which those baselines pay for in
  // unrepaired drift instead).
  std::ofstream rbf(outDir+"/rebuilds.tsv"); rbf<<"batch\n";
  std::ofstream dpf;  // hier mode: per-batch open-set DP stats (committed nodes / total)
  std::ofstream self; // atomic-selector prediction and post-commit objective validation
  if(emitHier){
    dpf.open(outDir+"/dp.tsv"); dpf<<"batch\tround\tn_dirty\tn_total\tdirty_frac\n";
    self.open(outDir+"/selection.tsv");
    self<<"batch\tround\tn_candidates\tn_selected\tpred_gain\tmarginal_cost\tpred_objective"
          "\tactual_gain\tactual_delta_f\ttrue_cost\ttrue_delta_f\taccepted\n";
  }

  // ------- initial detect (both modes) -------
  Hierarchy H0 = detect_iter(g, inst, 42);
  std::vector<int> C = H0.levels.back();          // top (communities) — fast-mode leaf partition
  std::vector<int> leaf;                           // hier-mode finest committed partition
  Hierarchy committedHierarchy;                    // hierarchy before the current batch

  // DYLO_INIT_PART: warm-start from an externally given leaf partition instead of cold-starting its own clustering.
  // Purpose: let DyLo take over an index already built by another system (e.g. GraphRAG's hierarchical_leiden),
  // after which only DyLo maintains it incrementally. Otherwise the two initial partitions are unrelated,
  // inheriting summaries by member set almost never hits, and the measured "re-summarization cost" is an artifact of the wiring rather than
  // a property of the algorithm. File format as com_*.txt: one community per line, tab-separated vertex ids.
  // Vertices not listed become singletons -- if upstream clusters only the largest connected component, the remaining vertices still need a home.
  auto load_init_part=[&](const char* path)->std::vector<int>{
    std::vector<int> memb(g.n, -1);
    std::ifstream in(path);
    if(!in){ fprintf(stderr,"[dylo] cannot open DYLO_INIT_PART: %s\n", path); exit(3); }
    std::string line; int cid=0; long placed=0;
    while(std::getline(in,line)){
      if(line.empty()) continue;
      std::istringstream ss(line); long v; bool any=false;
      while(ss>>v){
        if(v>=0 && v<(long)g.n){ memb[(size_t)v]=cid; placed++; any=true; }
      }
      if(any) cid++;
    }
    long singles=0;
    for(int i=0;i<g.n;i++) if(memb[i]<0){ memb[i]=cid++; singles++; }
    fprintf(stderr,"[dylo] warm-start partition: %d communities, %ld vertices assigned, %ld singletons added\n",
            cid, placed, singles);
    return memb;
  };

  if(emitHier){
    const char* ip = std::getenv("DYLO_INIT_PART");
    leaf = ip ? load_init_part(ip)
              : H0.levels[0];                      // maintain the FINEST level incrementally
    Hierarchy Hc = COMBINED ? build_levels_combined(g, leaf, inst, rng, TD_MAXSZ, HIER_GAMMA)
                 : TOPDOWN  ? build_levels_topdown(g, leaf, inst, rng, TD_MAXSZ)
                            : build_levels_from_leaf(g, leaf, inst, rng, HIER_GAMMA);
    leaf = TOPDOWN ? Hc.levels.back() : Hc.levels[0];
    cinH.clear();
    for(int v=0;v<g.n;v++){
      if(g.k[v]<=0) continue;
      cinH[leaf[v]] += 2.0*g.self[v];
      for(auto&[u,w]:g.adj[v]) if(leaf[(int)u]==leaf[v]) cinH[leaf[v]] += w;
    }
    committedHierarchy = Hc; costTreeValid=false;
    if(std::getenv("DYLO_DEBUG_FB")) fprintf(stderr,"[init] committedHierarchy.levels.size()=%zu\n",committedHierarchy.levels.size());
    // Q and the flat partition are reported at the COMMITTED LEAF level (the summary-tree
    // leaves = the communities A,B,C,D of the paper's model). Withholding forgoes a leaf's
    // re-optimization gain g(c) -> the leaf partition is stale -> Q(C') is lower, exactly as
    // F = Q - lambda*Cost prescribes. (Hc.Q is the re-optimized TOP, which would hide this.)
    const auto& rep0 = TOPDOWN ? Hc.levels.back() : Hc.levels[0];
    metrics<<"0\t0.000000\t"<<modularity(g,rep0)<<"\t"<<count_comm(rep0)<<"\t0\n"; metrics.flush();
    write_levels(0, Hc);
    write_com(0, rep0);   // was levels[0] -- the FINEST level under TOPDOWN, while every com_k (k>0) writes the coarse level
  } else {
    metrics<<"0\t0.000000\t"<<modularity(g,C)<<"\t"<<count_comm(C)<<"\t0\n"; metrics.flush();
    if(!noCom) write_com(0,C);
  }

  // ---- persistent state for the FAST optimized incremental path ----
  // dense community ids (no per-batch renumber), incremental community-degree table,
  // scratch neighbour-community histogram, margin-gated affected region.
  std::vector<double> ctotF;   // ctotF[c] = total degree of community c
  std::vector<double> kbuf;    // scratch: community -> incident weight from current vertex (0 between uses)
  std::vector<char>   inA;     // affected-region membership flag (reset per batch via the A list)
  std::vector<double> margin;  // margin[u] = gain slack: extra incident weight u needs to flip (>=0)
  std::vector<double> pert;    // pert[u]  = perturbation accrued since u's margin was last set
  int nextComm=0;
  if(!emitHier && !fromScratch){
    for(int v=0; v<g.n; v++) nextComm = std::max(nextComm, C[v]+1);
    ctotF.assign(nextComm, 0.0); for(int v=0;v<g.n;v++) ctotF[C[v]] += g.k[v];
    kbuf.assign(nextComm, 0.0); inA.assign(g.n, 0);
    margin.assign(g.n, -1.0);  // -1 => never evaluated -> always process on first touch
    pert.assign(g.n, 0.0);
    g.enable_fadj();           // cache-friendly flat adjacency for the hot histogram loop
  }

  // ---- L1 margin-CERTIFICATE gate for the HIER cost-aware path ----
  // ctotH: persistent community volume (unchanged from driver_lazy_incq.cxx).
  // kvComm[v][c]: EXACT edge weight from v to community c (replaces the
  // per-round `kin` histogram rebuild via a g.adj scan -- now O(1) lookup).
  // Certificate machinery (store + 3 lazy-deletion heaps) is a direct port
  // of driver_margin_certificate.cxx's validated design: see that file's
  // header comment for the full derivation (why k_v/k_{v,A}/k_{v,D} are
  // tracked exactly rather than via interval tolerances; why global M2
  // growing, community A's volume growing, AND community D's volume
  // SHRINKING are the three genuinely-dangerous drift directions).
  std::unordered_map<int,double> ctotH;
  std::vector<std::unordered_map<int,double>> kvComm;
  // marginH/pertH: scoped to WITHIN-A-ROUND cascade propagation only now
  // (a tentative mover's neighbours may need re-examination against the
  // round-local `fresh[]` state, which the persistent certificate system
  // does not track -- rebuilding proper certificates against tentative,
  // possibly-rejected state would be unsound). The certificate system
  // instead governs the two things it CAN reason about soundly: which
  // vertices seed a round in the first place (roundSeed, from the batch's
  // real committed edges) and which seed the NEXT round (nextSeed, from
  // this round's ACTUALLY COMMITTED moves) -- see the round-seeding and
  // post-commit blocks below.
  std::vector<double> marginH, pertH;
  struct CertH { int v=-1,D=-1; double M2_expiry=1e300, SigA_expiry=1e300, SigD_floor=-1e300,
                 Kv_expiry=1e300; long version=0; };
  std::vector<CertH> certStore;
  std::unordered_map<long long,int> vD2cert;
  std::vector<std::vector<int>> certsOfVertex;
  long certVersionCounter=0;
  // Smallest Kv_expiry among a vertex's live certificates: the degree at which
  // SOME certificate of v could first be invalidated by v's own degree growth.
  // Conservative -- dropping a certificate never raises it back -- so it can
  // only cause an unnecessary recheck, never a missed one.
  std::vector<double> kvExpMin;
  long statRecheckSkipped=0, statRecheckRun=0;
  using HEntry = std::tuple<double,int,long>;
  std::priority_queue<HEntry, std::vector<HEntry>, std::greater<HEntry>> m2HeapH;
  std::unordered_map<int, std::priority_queue<HEntry, std::vector<HEntry>, std::greater<HEntry>>> sigHeapH;
  std::unordered_map<int, std::priority_queue<HEntry>> sigDHeapH;
  // Epoch stamps, not flags: clearing two O(N) byte arrays once per batch was
  // a per-batch floor independent of how little the batch actually changed.
  // A vertex is "touched this batch" iff its stamp equals the current epoch,
  // so the clear is a single increment.
  std::vector<int> touchedEpoch, queueEpoch;
  int batchEpoch=0;

  auto certKeyH=[&](int v,int D)->long long{ return ((long long)v<<32) | (unsigned)D; };
  auto dropCertH=[&](int v,int D){
    long long key=certKeyH(v,D);
    auto it=vD2cert.find(key);
    if(it==vD2cert.end()) return;
    int cid=it->second;
    certStore[cid].version++;
    vD2cert.erase(it);
    auto& lst=certsOfVertex[v];
    for(size_t i=0;i<lst.size();++i) if(lst[i]==cid){ lst[i]=lst.back(); lst.pop_back(); break; }
  };
  auto computeHc=[&](int v,int A,int D)->double{
    double kv=g.k[v];
    double kvA = kvComm[v].count(A)?kvComm[v][A]:0.0;
    double kvD = kvComm[v].count(D)?kvComm[v][D]:0.0;
    double SigA=ctotH[A]-kv, SigD=ctotH[D];
    return g.m2*(kvD-kvA) - GAMMA_H*kv*(SigD-SigA+kv);
  };
  auto buildCertH=[&](int v,int A,int D,double H0){
    double kv=g.k[v];
    double kvA = kvComm[v].count(A)?kvComm[v][A]:0.0;
    double kvD = kvComm[v].count(D)?kvComm[v][D]:0.0;
    double mu0=-H0;
    bool mDangerous=(kvD>kvA);
    // H = m2*(kvD-kvA) - kv*(ctot[D] - ctot[A] + 2kv), so dH/dkv is
    //     -(ctot[D] - ctot[A] + 4kv),
    // positive -- i.e. the certificate weakens as v gains degree -- only when
    // ctot[A] - ctot[D] > 4kv: the destination has to be smaller than the
    // source by more than four times v's degree. Otherwise v's own growth
    // makes the certificate STRONGER and no recheck is ever needed for it.
    double kvSlope = ctotH[A] - ctotH[D] - 4.0*kv;
    bool kDangerous = (kvSlope > 1e-12);
    int nDim=(mDangerous?3:2) + (kDangerous?1:0);
    double share=0.97*mu0/nDim;
    long long key=certKeyH(v,D);
    int cid;
    auto it=vD2cert.find(key);
    if(it!=vD2cert.end()) cid=it->second;
    else { cid=(int)certStore.size(); certStore.push_back({}); vD2cert[key]=cid; certsOfVertex[v].push_back(cid); }
    CertH& c=certStore[cid];
    c.v=v; c.D=D; c.version=++certVersionCounter;
    c.M2_expiry = mDangerous ? (g.m2 + share/(kvD-kvA)) : 1e300;
    c.SigA_expiry = ctotH[A] + share/(GAMMA_H*kv);
    c.SigD_floor  = ctotH[D] - share/(GAMMA_H*kv);
    c.Kv_expiry   = kDangerous ? (kv + share/kvSlope) : 1e300;
    if((int)kvExpMin.size()<=v) kvExpMin.resize(v+1,1e300);
    kvExpMin[v] = std::min(kvExpMin[v], c.Kv_expiry);
    if(mDangerous) m2HeapH.push({c.M2_expiry, cid, c.version});
    sigHeapH[A].push({c.SigA_expiry, cid, c.version});
    sigDHeapH[D].push({c.SigD_floor, cid, c.version});
  };
  std::vector<int> queueH;
  long statVertexTriggers=0;   // total genuinely-new enqueue events across the whole run
  auto enqueueH=[&](int v){
    touchedEpoch[v]=batchEpoch;
    if(queueEpoch[v]!=batchEpoch){ queueEpoch[v]=batchEpoch; queueH.push_back(v); statVertexTriggers++; }
  };

  // ---- Level 2: bucket-level F-certificate (F = DeltaQ - lambda*Cost) ----
  // Every action (v,D) that Level 1 finds live (g>0) would, in the OLD
  // pure-Q design, trigger `v` immediately. Instead, route it into its
  // LCA bucket's accumulator; only flush the bucket (enqueue everyone who
  // contributed since the last flush) once the accumulated positive-gain
  // mass exceeds Gamma_b -- a STRUCTURAL, always-valid lower bound on what
  // ANY non-empty committed subset of this bucket must pay: committing any
  // action whose anchors' LCA is h_b unconditionally dirties h_b itself
  // (closure.hpp's child->parent(cap=inf) edges force every ancestor up to
  // root into the dirty set), so cost(S) >= weight(h_b) for S != empty,
  // giving F_b* <= max_S DeltaQ(S) - lambda*weight(h_b) <= 0 as long as
  // the accumulated positive-gain mass (a valid, if loose, upper bound on
  // max_S DeltaQ(S) since committing nothing is always feasible) stays
  // under Gamma_b = lambda*weight(h_b). D_b is intentionally NOT tightened
  // by removing stale contributions when an action's gain later drops back
  // to <=0 -- over-counting only makes the bound MORE conservative (safe),
  // matching "allow false positives, never allow a missed positive-F
  // action" (per this design's own stated principle).
  std::unordered_map<long long,double> nodeWeightPersist;   // (level,commId) -> weight (member/child count)
  std::unordered_map<int,int> repOfComm;                    // leaf commId -> a representative vertex
  std::unordered_map<long long,double> bucketD;              // bucketKey -> accumulated positive-gain mass
  std::unordered_map<long long,std::unordered_set<int>> bucketMembers;
  auto bkey=[&](int level,int comm)->long long{ return ((long long)level<<40) | (unsigned)comm; };
  // Virtual-root bucket key: real (level,commId) pairs use bkey(), which
  // packs level into bits [40:63) and a non-negative commId into [0:40) --
  // a single sentinel using level=63 (unreachable by any real level) can
  // never collide with a real bucket.
  auto findBucket=[&](int A,int D)->long long{
    auto ra=repOfComm.find(A), rd=repOfComm.find(D);
    if(ra==repOfComm.end()||rd==repOfComm.end()) return -1;
    int va=ra->second, vd=rd->second;
    int nlev=(int)committedHierarchy.levels.size();
    for(int li=0; li<nlev; li++){
      if(va>=(int)committedHierarchy.levels[li].size()||vd>=(int)committedHierarchy.levels[li].size())
        return -1;   // new-vertex sentinel or size mismatch -> caller flushes directly
      int ca=committedHierarchy.levels[li][va], cd=committedHierarchy.levels[li][vd];
      if(ca==cd) return bkey(li,ca);
    }
    // No shared ancestor within the REAL levels (the committed hierarchy is
    // a forest -- multiple top-level roots, e.g. from disconnected graph
    // components, or simply because build_levels_from_leaf found nothing
    // worth aggregating above the leaf level at all: confirmed via direct
    // inspection this session that committedHierarchy is frequently only
    // 1 level, i.e. EVERY leaf is its own top-level root). Route to a
    // single virtual super-root bucket rather than always-flushing: it
    // still costs weight 0 (nothing is genuinely dirtied by "the virtual
    // root" itself -- same reasoning as localized_closure.hpp's own
    // virtual-root fix earlier this session), so Gamma=0 there and any
    // action landing in it still fires immediately -- this fix changes
    // NOTHING about correctness or behavior for cross-top-level actions,
    // it only avoids treating them as a special "no bucket" case.
    return bkey(63, 0);
  };
  // B3 fix (DYLO_MARGINAL_GAMMA=1).
  // Review B3: the bucket-pruning test uses the absolute weight w(B), but Eq.(7) actually charges the
  // MARGINAL weight w_bar_t(B). For a destination already refreshed in this batch,
  // w_bar_t(B)=0 -- opening B is free, so g(B)-lambda*w(B) is a LOWER bound,
  // not the upper bound the paper claims, and "safe reduction" as written is false.
  // The paper gives its own counterexample (4-incAlgo.tex:230): after a commit its refresh closure enters
  // O_t, and every node on that path has zero marginal weight in later rounds of the batch.
  // The error is one-sided: over-pruning only loses value and never admits an unsound commit.
  // Here O_t is exposed to gammaOf: if the node is already in batchDirty then Gamma=0 and the bucket flushes immediately.
  const bool MARGINAL_GAMMA = [](){ const char* e=std::getenv("DYLO_MARGINAL_GAMMA");
                                    return e && std::atoi(e)!=0; }();
  const std::set<std::pair<int,int>>* curBatchDirty = nullptr;
  long statGammaTested=0, statGammaZeroed=0;   // B3 diagnostics: how many tests / how many of them had w_bar=0
  long statGammaNonEmptyOt=0;                  // how many of them had a non-empty O_t (could hit, timing-wise)
  long statOtMax=0;                            // the largest O_t seen
  auto gammaOf=[&](long long bk)->double{
    if(lamInf) return 1e300;
    if(bk==bkey(63,0)) return 0.0;   // virtual root: nothing genuinely dirtied, no deferral possible there
    if(curBatchDirty){
      statGammaTested++;
      if(!curBatchDirty->empty()) statGammaNonEmptyOt++;
      if((long)curBatchDirty->size()>statOtMax) statOtMax=(long)curBatchDirty->size();
      int lv=(int)(bk>>40), cm=(int)(bk & ((1LL<<40)-1));
      if(curBatchDirty->count({lv,cm})){
        statGammaZeroed++;
        if(MARGINAL_GAMMA) return 0.0;   // w_bar_t(B)=0 -> opening B is free
      }
    }
    auto it=nodeWeightPersist.find(bk);
    double w=(it!=nodeWeightPersist.end())?it->second:1.0;   // conservative fallback if not yet indexed
    return lambda*w;
  };
  // Replaces a direct enqueueH(v) whenever Level 1 found g=H/M2>0 for
  // action (v,D). Falls back to an immediate flush (matching this design's
  // own listed fallback rules: new vertex/community, action not mappable
  // to an existing bucket) when no LCA bucket can be found.
  const bool disableBucket = std::getenv("DYLO_DISABLE_BUCKET")!=nullptr;   // validation: degrade to always-on
  long statBucketAdd=0, statBucketFlush=0, statBucketFallback=0;
  auto gateLevel2=[&](int v,int D,double g){
    if(disableBucket){ enqueueH(v); return; }
    int A=leaf[v];
    long long bk=findBucket(A,D);
    if(bk<0){ statBucketFallback++; enqueueH(v); return; }
    statBucketAdd++;
    bucketD[bk]+=g;
    bucketMembers[bk].insert(v);
    if(bucketD[bk] > gammaOf(bk)){
      statBucketFlush++;
      for(int w: bucketMembers[bk]) enqueueH(w);
      bucketD[bk]=0.0;
      bucketMembers[bk].clear();
    }
  };
  auto directCheckH=[&](int v,int D)->bool{
    int A=leaf[v];
    if(D==A) return false;
    double H=computeHc(v,A,D);
    if(H>=-1e-12){ gateLevel2(v,D,H/g.m2); return true; }
    buildCertH(v,A,D,H); return false;
  };
  auto recheckAllH=[&](int v){
    int A=leaf[v];
    if((int)kvExpMin.size()<=v) kvExpMin.resize(v+1,1e300);
    kvExpMin[v]=1e300;                 // rebuilt by the buildCertH calls below
    statRecheckRun++;
    auto lst=certsOfVertex[v];
    for(int cid:lst){
      if(certStore[cid].v!=v) continue;
      int D=certStore[cid].D;
      double H=computeHc(v,A,D);
      if(H>=-1e-12){ gateLevel2(v,D,H/g.m2); dropCertH(v,D); }
      else buildCertH(v,A,D,H);
    }
  };
  auto sweepSigH=[&](int A){
    auto it=sigHeapH.find(A);
    if(it==sigHeapH.end()) return;
    auto& h=it->second;
    while(!h.empty()){
      auto [exp,cid,ver]=h.top();
      if(exp>ctotH[A]) break;
      h.pop();
      if(certStore[cid].version!=ver) continue;
      int v=certStore[cid].v, D=certStore[cid].D;
      if(leaf[v]!=A) continue;
      double H=computeHc(v,A,D);
      if(H>=-1e-12){ gateLevel2(v,D,H/g.m2); dropCertH(v,D); }
      else buildCertH(v,A,D,H);
    }
  };
  auto sweepSigDH=[&](int D){
    auto it=sigDHeapH.find(D);
    if(it==sigDHeapH.end()) return;
    auto& h=it->second;
    while(!h.empty()){
      auto [floor,cid,ver]=h.top();
      if(floor<=ctotH[D]) break;
      h.pop();
      if(certStore[cid].version!=ver) continue;
      int v=certStore[cid].v, Dc=certStore[cid].D;
      if(Dc!=D) continue;
      int A=leaf[v];
      double H=computeHc(v,A,Dc);
      if(H>=-1e-12){ gateLevel2(v,Dc,H/g.m2); dropCertH(v,Dc); }
      else buildCertH(v,A,Dc,H);
    }
  };
  if(emitHier){
    for(int u=0; u<g.n; u++) ctotH[leaf[u]] += g.k[u];
    kvComm.assign(g.n, {});
    for(int v=0; v<g.n; v++) for(auto&[u,w]: g.adj[v]) kvComm[v][leaf[u]] += w;
    certsOfVertex.assign(g.n, {});
    marginH.assign(g.n, -1.0);
    pertH.assign(g.n, 0.0);
    // Seed initial certificates for every action already present in the
    // starting partition (see driver_margin_certificate.cxx's header
    // comment on why this is required for correctness, not optional).
    for(int v=0; v<g.n; v++){
      int A=leaf[v];
      for(auto&[D,w]:kvComm[v]){
        if(D==A) continue;
        double H=computeHc(v,A,D);
        if(H<-1e-12) buildCertH(v,A,D,H);
      }
    }
  }

  int batchesSinceRebuild=0, nRebuild=0, nAudit=0, nReclaim=0, nPeel=0; long nReclaimed=0;
  // L3, ski-rental form (thm:competitive). The snapshot rule "rebuild when the
  // instantaneous gap exceeds R" never fires if drift stays just under R at
  // every audit, however long the stream. The competitive argument is instead
  // about ACCUMULATED carried drift: staying stale pays rent every batch, and
  // you buy (rebuild) once the rent since the last purchase reaches its price.
  double driftAcc=0.0, gapAtLastRebuild=0.0;
  long statTotalEdges=0;
  // Wall-clock budget for the STREAM only. Building the initial graph and its
  // hierarchy is a one-off that scales with |E| and would otherwise consume the
  // whole allowance on the largest graphs -- on LiveJournal's 38.6M initial
  // edges it takes most of an hour, leaving a 3% sample of the stream. The
  // clock therefore starts here, after the initial index exists.
  // Per-batch modularity is instrumentation, not maintenance: it is already
  // outside the timed region, but on the largest graph it is an O(N+|E|) walk
  // costing ~19s per batch against 21ms of actual work, so it consumes the
  // stream budget and caps coverage at 4% of the stream. With DYLO_Q_AT_END the
  // metrics line records Q only for the final batch, which is the only one the
  // quality tables read anyway.
  const bool Q_AT_END = std::getenv("DYLO_Q_AT_END")!=nullptr;
  const double STREAM_BUDGET = [](){ const char* e=std::getenv("DYLO_STREAM_BUDGET_S"); return e?std::atof(e):0.0; }();
  const auto streamT0 = std::chrono::high_resolution_clock::now();
  int batchesRun = 0;
  for(int k=1;k<=nB;k++){
    if(STREAM_BUDGET>0.0){
      double el=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-streamT0).count();
      if(el>=STREAM_BUDGET){
        fprintf(stderr,"[dylo:%s] stream budget %.0fs reached after %d/%d batches\n",
                algo.c_str(),STREAM_BUDGET,batchesRun,nB);
        break;
      }
    }
    batchesRun++;
    // Parsing the batch file is I/O, not maintenance, and is deliberately
    // OUTSIDE the timed region -- as are the community files written at the end
    // of the batch and the modularity computed only to fill in metrics.tsv.
    // Applying the parsed batch to the graph IS maintenance and stays inside:
    // every incremental baseline has to do the same work.
    std::vector<std::pair<uint32_t,uint32_t>> raw;
    read_edges(extend(incBase,"_"+std::to_string(k)),
               [&](uint32_t u,uint32_t v,double){ raw.emplace_back(u,v); });
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> seed;
    std::vector<std::pair<uint32_t,uint32_t>> ins;       // inserted edges this batch (for margin gate)
    seed.reserve(raw.size()*2); ins.reserve(raw.size());
    for(auto&[u,v] : raw){ g.add_edge(u,v,1.0); seed.push_back(u); seed.push_back(v); ins.emplace_back(u,v); }
    statTotalEdges += (long)ins.size();

    if(fromScratch){
      Hierarchy Hk = detect_iter(g, inst, 42);
      C = Hk.levels.back();
      // The summary hierarchy is built from the detected partition by exactly the
      // rule the incremental path uses, so a cost comparison between the two
      // isolates the partition's churn rather than the way its tree was made.
      // It sits inside the timed region because a from-scratch pipeline really
      // does have to rebuild it.
      Hierarchy Hfs = build_levels_from_leaf(g, C, inst, rng, HIER_GAMMA);
      auto t1=std::chrono::high_resolution_clock::now();
      metrics<<k<<"\t"<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\t"
             <<modularity(g,C)<<"\t"<<count_comm(C)<<"\t0\n"; metrics.flush();
      if(!noCom){ write_levels(k, Hfs); write_com(k,C); }
      continue;
    }

    if(emitHier){
      // ---- MULTI-ROUND cost-aware hierarchical update (Algorithm 1) ----
      // Repeat the per-round logic (propagate, select atomic actions, commit-or-reject)
      // until a round is rejected (DeltaF<=0) or finds nothing to move, seeding each new
      // round from the previous round's movers' NEIGHBOURS (their local landscape just
      // changed, which may open new positive-DeltaF moves the movers themselves won't
      // re-trigger) -- the same enqueue rule already used within a single round's own
      // propagation. Capped at MAX_ROUNDS as a safety net (empirically observed max on
      // the efficiency-tier datasets: 4 rounds). Matches the paper's termination argument
      // (every retained round strictly increases F; finitely many partitions).
      const int MAX_ROUNDS = 10;
      int roundsRun = 0;
      bool anyRoundAccepted = false;

      // For new-vertex bookkeeping we still need the pre-batch leaf size once.
      const int oldN0=(int)leaf.size();
      for(int i=(int)leaf.size();i<g.n;i++) leaf.push_back(i+1000000000);
      // (1) size-prefilter state. Built once from the initial partition, then
      // followed: arrivals add their sentinel here, commits move one unit, and
      // the paths that renumber `leaf` wholesale clear it for a fresh build.
      for(int v=oldN0; v<g.n; v++) pendingTreeV.push_back(v);
      for(auto&[a,b] : ins){
        if(a<(uint32_t)g.n) pendingTreeV.push_back((int)a);
        if(b<(uint32_t)g.n) pendingTreeV.push_back((int)b);
      }
      if((int)inCommSize.size()<g.n) inCommSize.resize(g.n,0);
      if(commSize.empty()){
        std::fill(inCommSize.begin(),inCommSize.end(),0);
        for(int v=0;v<oldN0 && v<g.n;v++) if(g.k[v]>0){ commSize[leaf[v]]++; inCommSize[v]=1; }
      }
      auto countOnce=[&](int v){ if(v<g.n && !inCommSize[v] && g.k[v]>0){ commSize[leaf[v]]++; inCommSize[v]=1; } };
      for(int i=oldN0;i<g.n;i++) countOnce(i);
      for(auto&[a,b] : ins){ countOnce((int)a); countOnce((int)b); }
      if(fineSize.empty()){
        const auto& L0=committedHierarchy.levels[0];
        for(int v=0;v<(int)L0.size() && v<g.n;v++) if(g.k[v]>0) fineSize[L0[v]]++;
      }
      // (2) the batch's actual re-summarization set: what the LLM is asked to
      // regenerate before this batch is reported committed. Accumulated across
      // rounds as (level, community) and priced once at the end of the batch.
      std::set<std::pair<int,int>> batchDirty;
      curBatchDirty = &batchDirty;      // B3: let gammaOf look up O_t
      bool batchAllDirty=false;
      // Fine communities dirtied by reclamation. Collected as raw ids and
      // closed over ancestors at the end of the batch, once the tree that
      // prices them is the one reclamation left behind.
      std::vector<int> reclaimDirtyFine;
      if((int)kvComm.size()<g.n){ kvComm.resize(g.n); certsOfVertex.resize(g.n); }
      if((int)marginH.size()<g.n){ marginH.resize(g.n,-1.0); pertH.resize(g.n,0.0); }
      ++batchEpoch;
      if((int)touchedEpoch.size()<g.n){ touchedEpoch.resize(g.n,-1); queueEpoch.resize(g.n,-1); }
      queueH.clear();

      // Level 2: refresh the persistent (level,commId)->weight index and
      // leaf-community representatives. Only needs rebuilding when the
      // HIERARCHY STRUCTURE changed (a round committed, or L3 rebuilt) --
      // raw edge insertion alone never touches committedHierarchy -- but
      // rebuilding once per batch (O(N), bounded, not per-edge) is simpler
      // and safe to always do here rather than tracking a dirty flag.
      {
        // nodeWeightPersist is exactly the cost tree's per-node weight, keyed
        // by (level, community) instead of by node id, so deriving it from the
        // tree removes a second O(N * levels) pass that recomputed the same
        // thing. Profiled at 23.8-24.7% of per-batch time once the hierarchy
        // became genuinely multi-level.
        //
        // It also fixes a key mismatch: the old code stored level 0 under
        // bkey(0, leaf[v]), but `leaf` is the MAINTAINED partition, which under
        // the top-down hierarchy is levels.BACK(), while findBucket looks up
        // levels[0] -- the finest split level. Every level-0 lookup therefore
        // missed and gammaOf silently fell back to w=1.0.
        // Rebuilt only when the hierarchy actually changed. The comment above
        // called the unconditional rebuild "simpler and safe"; it is also the
        // dominant per-batch cost on large graphs, because it does one hash
        // insert per VERTEX regardless of how little the batch touched --
        // measured as a floor of ~0.1 ms per thousand vertices, i.e. most of
        // DBLP's per-batch time. `costTreeValid` already marks exactly the
        // events that invalidate it (a committed round, a reclamation, a
        // rebuild), so it is the right trigger.
        if(!costTreeValid){
          costTreeCache=make_cost_tree(g,committedHierarchy); costTreeValid=true;
          nodeWeightPersist.clear(); repOfComm.clear();
          for(int v=0; v<g.n; v++) if(g.k[v]>0) repOfComm[leaf[v]]=v;
          for(int li=0; li<(int)costTreeCache.ids.size(); li++)
            for(const auto&[c,id] : costTreeCache.ids[li])
              nodeWeightPersist[bkey(li,c)] = costTreeCache.weight[id];
        }
        // Arrivals and vertices gaining their first edge are not covered by a
        // skipped rebuild, so their communities get a representative here --
        // O(batch), not O(N).
        for(auto&[u,v] : ins){
          if(u<(uint32_t)g.n && g.k[u]>0) repOfComm[leaf[u]]=(int)u;
          if(v<(uint32_t)g.n && g.k[v]>0) repOfComm[leaf[v]]=(int)v;
        }
      }

      // L1 (certificate scheme): update the persistent community volume and
      // EXACT kvComm for this batch's raw edge insertions, run direct
      // checks + full recheck on both endpoints (k_v changed for both --
      // see driver_margin_certificate.cxx's header comment on why this
      // must cover ALL of a touched vertex's certs, not just the specific
      // newly-touched target). New vertices are still sentinel-leaf until
      // touched by their first real edge -- leaf[u]!=leaf[v] is then
      // trivially true so directCheckH/recheckAllH run for them exactly as
      // for any other cross-community insertion.
      for(auto&[u,v]:ins){
        if(u==v){ ctotH[leaf[u]] += 2.0; cinH[leaf[u]] += 2.0; continue; }
        int cu=leaf[u], cv=leaf[v];
        ctotH[cu] += 1.0; ctotH[cv] += 1.0;
        if(cu==cv) cinH[cu] += 2.0;                 // an intra-community insertion
        kvComm[u][cv]+=1.0; kvComm[v][cu]+=1.0;
        if(cu!=cv){ directCheckH((int)u,cv); directCheckH((int)v,cu); }
        // GATED RECHECK (DYLO_LAZY_CERT, default on). Rechecking every
        // certificate of both endpoints on every insertion was profiled at
        // 72-81% of L1 and 17-25% of total time, and it is mostly wasted: an
        // insertion raises k_v, and by the Kv dimension above that can only
        // invalidate certificates whose destination is smaller than the source
        // by more than 4*k_v. kvExpMin[v] is the degree at which the FIRST of
        // v's certificates could reach that point, so below it there is
        // provably nothing to find. The other three drift dimensions (m2,
        // Sigma_A, Sigma_D) are unaffected -- they keep their own heaps, which
        // are swept once per batch as before.
        auto recheckGatedH=[&](int x){
          if(!LAZY_CERT){ recheckAllH(x); return; }
          if(x<(int)kvExpMin.size() && g.k[x] < kvExpMin[x]){ statRecheckSkipped++; return; }
          recheckAllH(x);
        };
        recheckGatedH((int)u); recheckGatedH((int)v);
      }
      // Global M2 / per-community drift sweeps (once per batch, matching
      // the FAST-mode port's own cadence rationale).
      while(!m2HeapH.empty()){
        auto [exp,cid,ver]=m2HeapH.top();
        if(exp>g.m2) break;
        m2HeapH.pop();
        if(certStore[cid].version!=ver) continue;
        int v=certStore[cid].v, D=certStore[cid].D, A=leaf[v];
        double H=computeHc(v,A,D);
        if(H>=-1e-12){ gateLevel2(v,D,H/g.m2); dropCertH(v,D); }
        else buildCertH(v,A,D,H);
      }
      { std::vector<int> touchedC;
        for(auto&[u,v]:ins){ if(u!=v){ touchedC.push_back(leaf[u]); touchedC.push_back(leaf[v]); } }
        for(int A:touchedC) sweepSigH(A);
      }

      HierCostTree& costTree = costTreeCache;
      std::vector<uint32_t> roundSeed(queueH.begin(), queueH.end());

      for(; roundsRun<MAX_ROUNDS; roundsRun++){
        if(roundSeed.empty()) break;
        if(!costTreeValid){ costTree=make_cost_tree(g,committedHierarchy); costTreeValid=true; }
        // These two maps were rebuilt from scratch every round, at one hash
        // insert per VERTEX -- profiled at 1251 of the round phase's 1300 ms on
        // DBLP, i.e. 44% of the whole run. Both already exist:
        //   community -> its cost-tree node  IS  costTree.ids[0], because under
        //     the aggregated hierarchy level 0 is the maintained partition;
        //   community -> a member vertex     IS  repOfComm, maintained
        //     incrementally for the audit and the tree patch.
        // Under the top-down hierarchy level 0 is the FINE split level, not the
        // maintained partition, and members of one maintained community do not
        // share a level path, so the representative must still be the one this
        // loop picks -- that mode keeps the original build.
        std::unordered_map<int,int> builtLeafNode, builtLeafRep;
        if(TOPDOWN){
          int refN = roundsRun==0 ? oldN0 : (int)leaf.size();
          for(int v=0;v<refN && v<(int)costTree.vertexLeaf.size();v++)
            if(costTree.vertexLeaf[v]>=0){
              builtLeafNode.emplace(leaf[v],costTree.vertexLeaf[v]);
              builtLeafRep.emplace(leaf[v],v);
            }
        }
        const std::unordered_map<int,int>& oldLeafNode =
          (!TOPDOWN && !costTree.ids.empty()) ? costTree.ids[0] : builtLeafNode;
        const std::unordered_map<int,int>& oldLeafRepresentative =
          TOPDOWN ? builtLeafRep : repOfComm;

        std::vector<int> fresh=leaf;
        struct MoveAction { int vertex,from,to; double gain; };
        std::vector<MoveAction> moves;

        // L1: ctot starts from the PERSISTENT ctotH (O(#communities) copy,
        // not an O(N) full-vertex rebuild) -- this round's tentative moves
        // mutate the local copy only; ctotH itself is updated after commit,
        // from the ACTUALLY selected moves (see below), since a rejected
        // tentative move must not leak into the persistent state.
        std::unordered_map<int,double> ctot = ctotH;
        std::vector<char> inA(g.n,0); std::vector<int> A;
        auto bumpH=[&](uint32_t x,double w){ if(x<(uint32_t)pertH.size()) pertH[x]+=w; };
        auto enqH=[&](uint32_t x){
          if(x<(uint32_t)g.n && !inA[x] && pertH[x] > marginH[x]){ inA[x]=1; A.push_back((int)x); }
        };
        for(uint32_t v:roundSeed) enqH(v);
        size_t head=0;
        while(head<A.size()){
          int u=A[head++], ci=fresh[u]; double ku=g.k[u];
          std::unordered_map<int,double> kin;
          for(auto&[v,w]:g.adj[u]) kin[fresh[v]]+=w;
          ctot[ci]-=ku;
          double stay=(kin.count(ci)?kin[ci]:0.0)-ctot[ci]*ku/g.m2;
          int best=ci; double bestgain=stay, second=-1e300;
          for(auto&[c,kic]:kin){
            if(c==ci) continue;
            double gn=kic-ctot[c]*ku/g.m2;
            if(gn>bestgain+1e-12){ second=bestgain; bestgain=gn; best=c; }
            else if(gn>second) second=gn;
          }
          ctot[best]+=ku;
          // L1 margin bookkeeping: gap between the chosen option and the
          // runner-up is how much MORE perturbation u can absorb before its
          // decision could possibly flip again (Lemma margin).
          marginH[u] = (second > -1e299) ? std::max(0.0, bestgain-second) : 1e18;
          pertH[u] = 0.0;
          if(best!=ci){
            // (1) SIZE PREFILTER. Moving u into `best` dirties `best`, whose
            // re-summarization is proportional to its member count, so a gain
            // that cannot cover lambda*|best| cannot survive the selector
            // either. Killing it here also suppresses the neighbour bump and
            // enqueue below, which is where most of the saving comes from --
            // a rejected candidate otherwise drags its whole neighbourhood
            // into the next round's frontier for nothing.
            //
            // The arrival credit is applied on the same terms as at the
            // action-build site: parking a newly-arrived vertex in its own
            // sentinel community is free in the cost model but real in the
            // pipeline, so the merge is credited the alpha+1 it avoids.
            // Without this the prefilter would park arrivals even harder
            // than the selector does.
            double gPrune = bestgain-stay;
            if(dylo_alpha()>0.0 && ci>=1000000000) gPrune += lambda*(dylo_alpha()+1.0);
            int dstSz = 0;
            if(PRUNE){
              if(PRUNE_LEAF){
                // the fine leaf u would join is the one holding its strongest
                // neighbour already inside `best` -- the same rule the action
                // build uses, so the prefilter and the anchor agree by
                // construction
                double bw=-1.0; int bu=-1;
                for(auto&[v2,w2]:g.adj[u]){
                  if((int)v2==u) continue;
                  if((int)v2<(int)fresh.size() && fresh[(int)v2]==best && w2>bw){ bw=w2; bu=(int)v2; }
                }
                const auto& L0=committedHierarchy.levels[0];
                if(bu>=0 && bu<(int)L0.size()){
                  auto it=fineSize.find(L0[bu]); dstSz = (it==fineSize.end())?1:it->second;
                } else dstSz=1;
              } else {
                auto it=commSize.find(best); dstSz = (it==commSize.end())?1:it->second;
              }
            }
            if(PRUNE && !lamInf && lambda>0.0 && gPrune - lambda*(double)dstSz < 0.0){
              statPruned++;
            } else {
              moves.push_back({u,ci,best,bestgain-stay});
              fresh[u]=best;
              for(auto&[v,w]:g.adj[u]) if(v<(uint32_t)g.n){ bumpH(v,w); enqH(v); }
            }
          }
        }

        { double vol=0.0; for(int x : A) vol += g.k[x];
          statVolA += vol/std::max(1.0,g.m2); statRegionBatches++; }
        if(moves.empty()) break;   // nothing to move -> loop halts (local optimum)

        std::unordered_map<int,int> tdDest;   // vertex -> chosen destination representative
        std::vector<std::pair<int,int>> srcOf;   // (action index, deferred source node)
        std::vector<AtomicAction> actions;
        std::vector<AtomicAction> trueActions;
        std::vector<int> actionMove;
        const double ALPHA=dylo_alpha();
        for(int mi=0;mi<(int)moves.size();mi++){
          const auto& mv=moves[mi];
          AtomicAction truth; truth.gain=mv.gain;
          // Fixed per-summary overhead, applied where it actually bites.
          // A newly-arrived vertex sits in a sentinel community that
          // make_cost_tree never sees (it only walks committedHierarchy),
          // so "leave the new vertex as its own singleton community" costs
          // EXACTLY ZERO in the model while "merge it into D" pays D's whole
          // ancestor closure. That asymmetry -- not lambda -- is what lets a
          // large lambda park every arrival in a trivial community for free
          // (up to 77.7% of communities on Epinions) with Q barely moving.
          // Reality: the arrival needs a summary either way. Parking costs
          // alpha+1 (a fresh one-member summary); merging pays closure(D)
          // but SAVES that alpha+1. Since F = gain - lambda*cost, crediting
          // the avoided alpha+1 to the merge action's gain is exactly
          // equivalent to charging it to the parking alternative, and keeps
          // every cost weight non-negative so the min-cut solver is
          // untouched. (Seeding tree-node weights with alpha, the obvious
          // first attempt, provably cannot work here: the singleton is not
          // a tree node at all, so that only made merging dearer -- verified
          // empirically, singleton count did not move for any alpha.)
          if(ALPHA>0.0 && mv.from>=1000000000) truth.gain += lambda*(ALPHA+1.0);
          int srcNode=-1, dstNode=-1;
          if(TOPDOWN){
            if(mv.vertex<(int)costTree.vertexLeaf.size()) srcNode=costTree.vertexLeaf[mv.vertex];
            // strongest neighbour already inside the destination decides which
            // fine leaf the vertex joins; O(deg) and reuses this adjacency walk
            double bw=-1.0; int bu=-1;
            for(auto&[u,w]:g.adj[mv.vertex]){
              if((int)u==mv.vertex) continue;
              if((int)u<(int)leaf.size() && leaf[(int)u]==mv.to && w>bw){ bw=w; bu=(int)u; }
            }
            if(bu>=0 && bu<(int)costTree.vertexLeaf.size()) dstNode=costTree.vertexLeaf[bu];
            tdDest[mv.vertex]=bu;
          } else {
            auto src=oldLeafNode.find(mv.from), dst=oldLeafNode.find(mv.to);
            if(src!=oldLeafNode.end()) srcNode=src->second;
            if(dst!=oldLeafNode.end()) dstNode=dst->second;
          }
          // (2) TRANSIENT OVERLAP. With deferral the source is NOT an anchor:
          // A keeps listing v until its residuals reach THETA, so this action
          // dirties the destination path alone. One anchor per action is what
          // makes the action set decomposable over the tree -- the per-node
          // gains stop depending on which sibling branch another action
          // happened to touch -- so the DP applies without the union-find
          // decomposability check having to certify it first.
          if(srcNode>=0 && !DEFER) truth.anchors.push_back(srcNode);
          if(dstNode>=0) truth.anchors.push_back(dstNode);
          // Deferring the source removes it as an ANCHOR but not as a COST: the
          // summary still has to be rewritten eventually, and a selector that
          // ignores that under-prices every move and accepts too many (measured
          // on Epinions: 17 committed rounds become 22 and re-summarization cost
          // rises 33%). The charge is put on the action's gain rather than on a
          // second anchor, which keeps one anchor per action -- and therefore
          // the tree DP -- while restoring the price. Actions leaving the same
          // community in the same round share the charge, because that
          // community is rewritten once however many members leave it.
          std::sort(truth.anchors.begin(),truth.anchors.end());
          truth.anchors.erase(std::unique(truth.anchors.begin(),truth.anchors.end()),
                              truth.anchors.end());
          if(truth.anchors.empty() || (mv.to>=1000000000 && dstNode<0)) continue;
          AtomicAction model=truth;
          if(legacySelector){
            if(dstNode<0) continue;
            model.anchors.assign(1,dstNode);
          }
          // Recorded AFTER the two `continue`s: taking the index before them
          // shifts every later entry of srcOf by one for each skipped candidate,
          // so the deferred-source charge lands on the wrong action, and when the
          // LAST candidate of a round is skipped the index is one past the end.
          // The out-of-range write corrupts the heap; it surfaces as a malloc
          // assertion only once the run is long enough to hit it (8,923 batches
          // at |dE|=10, never at |dE|=1000).
          if(DEFER && SRC_CHARGE && srcNode>=0) srcOf.push_back({(int)trueActions.size(), srcNode});
          actions.push_back(std::move(model));
          trueActions.push_back(std::move(truth));
          actionMove.push_back(mi);
        }
        if(actions.empty()) break;

        if(DEFER && SRC_CHARGE && !srcOf.empty()){
          std::unordered_map<int,int> leavers;               // source node -> movers this round
          for(auto&[ai,sn] : srcOf) leavers[sn]++;
          std::unordered_map<int,double> closure;            // source node -> its ancestor closure
          // What the deferred source will actually cost depends on whether it
          // survives. A community that loses its last member is DELETED, not
          // rewritten: the summary disappears and what changes is its parent's
          // child set, so the price is the parent's closure, not its own. That
          // is the common case here -- 99.6-100% of committed moves empty their
          // source -- so charging the source's own closure over-prices the
          // frequent case and under-prices nothing.
          for(auto&[sn,cnt] : leavers){
            int start = sn;
            if(sn<(int)costTree.weight.size() && costTree.weight[sn] <= (double)cnt + 1e-9)
              start = costTree.parent[sn];        // the source empties -> charge its parent
            double c=0.0;
            for(int x=start; x>=0 && x<(int)costTree.weight.size(); x=costTree.parent[x]) c+=costTree.weight[x];
            closure[sn]=c;
          }
          for(auto&[ai,sn] : srcOf){
            const double share = closure[sn] / (double)leavers[sn];
            actions[ai].gain     -= lambda*share;
            trueActions[ai].gain -= lambda*share;
          }
        }

        // SELECTION. Two anchors per action -- the vertex's old community and its
        // new one -- make the actions couple across the tree, and the exact
        // optimum then needs a max-weight-closure/min-cut solve. Deferring the
        // source (Section transient overlap) leaves ONE anchor per action, and
        // the problem collapses to the ancestor-closed tree DP of
        // Theorem~\ref{thm:dp}: open(c) = g(c) - lambda*w(c) + sum of positive
        // children, dirty iff open >= 0. Same optimum, no flow solve.
        //
        // The DP is used whenever every action is single-anchored, which the
        // default configuration guarantees; the min-cut path stays for the
        // two-anchor case (DYLO_DEFER=0) so both remain reachable and testable.
        bool allSingle = true;
        for(const auto& a : actions) if(a.anchors.size()!=1){ allSingle=false; break; }
        LocalizedSelection chosen;
        if(!lamInf){
          if(allSingle && USE_TREE_DP){
            // one DP node per cost-tree node that some action can reach, plus
            // its ancestors; gains accumulate on the anchor, weights come from
            // the cost tree
            std::unordered_map<int,int> nid;                 // tree node -> dp node
            std::vector<int> back;
            auto idOf=[&](int c)->int{
              auto it=nid.find(c); if(it!=nid.end()) return it->second;
              int k2=(int)back.size(); nid[c]=k2; back.push_back(c); return k2;
            };
            for(const auto& a : actions)
              for(int c=a.anchors[0]; c>=0; c=costTree.parent[c]) idOf(c);
            DPTree T; const int NN=(int)back.size();
            T.children.assign(NN,{}); T.g.assign(NN,0.0); T.w.assign(NN,0.0);
            for(int i=0;i<NN;i++) T.w[i]=costTree.weight[back[i]];
            for(int i=0;i<NN;i++){
              int par=costTree.parent[back[i]];
              auto it = (par>=0) ? nid.find(par) : nid.end();
              if(it!=nid.end()) T.children[it->second].push_back(i);
              else T.roots.push_back(i);
            }
            for(const auto& a : actions) T.g[nid[a.anchors[0]]] += a.gain;
            std::vector<char> d;
            if(GREEDY){
              // each destination with gain decides independently: it pays its whole ancestor chain, shared with no sibling.
              d.assign(NN, 0);
              for(int i=0;i<NN;i++){
                if(T.g[i] <= 0.0) continue;
                double own = 0.0;
                for(int c=back[i]; c>=0; c=costTree.parent[c]) own += costTree.weight[c];
                if(T.g[i] - lambda*own >= 0.0)
                  for(int c=i; c>=0; ){            // if accepted, close over ancestors
                    d[c]=1;
                    int par=costTree.parent[back[c]];
                    auto it=(par>=0)?nid.find(par):nid.end();
                    c = (it!=nid.end()) ? it->second : -1;
                  }
              }
            } else {
              d = open_set_dp(T, lambda);
            }
            chosen.dirty_nodes.assign(costTree.parent.size(),0);
            for(int i=0;i<NN;i++) if(d[i]) chosen.dirty_nodes[back[i]]=1;
            chosen.selected_actions.assign(actions.size(),0);
            chosen.selected_gain=0.0; chosen.marginal_cost=0.0;
            for(size_t ai=0; ai<actions.size(); ai++)
              if(d[nid[actions[ai].anchors[0]]]){
                chosen.selected_actions[ai]=1;
                chosen.selected_gain += actions[ai].gain;
              }
            for(int i=0;i<NN;i++) if(d[i]) chosen.marginal_cost += T.w[i];
            chosen.objective = chosen.selected_gain - lambda*chosen.marginal_cost;
          } else {
            chosen=select_atomic_actions_localized(costTree.parent,costTree.weight,actions,lambda);
          }
        }
        std::vector<int> candidate=leaf;
        int nSelected=0;
        for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
          if(!chosen.selected_actions[ai]) continue;
          nSelected++;
          const auto& mv=moves[actionMove[ai]];
          candidate[mv.vertex]=mv.to;
        }

        // O(vol(selected)+|touched communities|) incremental Delta(2m*Q),
        // replacing the two full-graph modularity() calls. EXACT (not an
        // approximation) for the touched communities' null-model term
        // (their Stot deltas are known exactly from the selected moves);
        // the internal-edge term is recomputed by scanning each SELECTED
        // move's vertex adjacency against the FINAL `candidate` assignment
        // (one O(deg(u)) scan per moved vertex -- self-loop contributions
        // exactly cancel in the from+to sum, verified algebraically against
        // detector.hpp's modularity() formula).
        double actualGain_inc;
        {
          std::unordered_set<int> touchedC;
          for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
            if(!chosen.selected_actions[ai]) continue;
            const auto& mv=moves[actionMove[ai]];
            touchedC.insert(mv.from); touchedC.insert(mv.to);
          }
          std::unordered_map<int,double> totBefore, totAfter;
          for(int c: touchedC) totBefore[c] = ctotH.count(c) ? ctotH[c] : 0.0;
          totAfter = totBefore;
          for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
            if(!chosen.selected_actions[ai]) continue;
            const auto& mv=moves[actionMove[ai]];
            double ku=g.k[mv.vertex];
            totAfter[mv.from]-=ku; totAfter[mv.to]+=ku;
          }
          double nullDelta=0.0;
          for(int c: touchedC) nullDelta += (totAfter[c]*totAfter[c] - totBefore[c]*totBefore[c]);
          double inDelta=0.0;
          for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
            if(!chosen.selected_actions[ai]) continue;
            const auto& mv=moves[actionMove[ai]];
            int u=mv.vertex; double toC=0.0, toD=0.0;
            for(auto&[v,w]:g.adj[u]){
              if((int)v==u) continue;
              int cv=candidate[v];
              if(cv==mv.from) toC+=w;
              if(cv==mv.to)   toD+=w;
            }
            inDelta += 2.0*(toD-toC);
          }
          actualGain_inc = inDelta/2.0 - nullDelta/(2.0*g.m2);
        }
        const bool VALIDATE_INC_Q = std::getenv("DYLO_VALIDATE_INCQ")!=nullptr;
        double actualGain = actualGain_inc;
        if(VALIDATE_INC_Q){
          const double qBefore=modularity(g,leaf);
          const double qAfter=modularity(g,candidate);
          const double actualGain_exact=(qAfter-qBefore)*g.m2/2.0;
          fprintf(stderr,"[incQ-validate] batch=%d round=%d inc=%.10f exact=%.10f diff=%.3e\n",
                  k, roundsRun, actualGain_inc, actualGain_exact, actualGain_inc-actualGain_exact);
        }
        std::vector<char> trueDirty(costTree.parent.size(),0);
        for(int ai=0;ai<(int)chosen.selected_actions.size();ai++) if(chosen.selected_actions[ai])
          for(int c:trueActions[ai].anchors)
            while(c>=0 && !trueDirty[c]){ trueDirty[c]=1; c=costTree.parent[c]; }
        double trueCost=0.0;
        for(int c=0;c<(int)trueDirty.size();c++) if(trueDirty[c]) trueCost+=costTree.weight[c];
        const double decisionCost=legacySelector?chosen.marginal_cost:trueCost;
        auto net_objective=[&](double cost){
          if(lamInf) return cost>0.0 ? -std::numeric_limits<double>::infinity() : actualGain;
          return actualGain-lambda*cost;
        };
        const double actualDeltaF=net_objective(decisionCost);
        const double trueDeltaF=net_objective(trueCost);
        const bool accepted=nSelected>0 && actualDeltaF>1e-12;

        self<<k<<"\t"<<roundsRun<<"\t"<<actions.size()<<"\t"<<nSelected<<"\t"
            <<chosen.selected_gain<<"\t"<<chosen.marginal_cost<<"\t"<<chosen.objective<<"\t"
            <<actualGain<<"\t"<<actualDeltaF<<"\t"<<trueCost<<"\t"<<trueDeltaF<<"\t"
            <<(accepted?1:0)<<"\n"; self.flush();

        // LOCAL-OPTIMALITY REPAIR (DYLO_PEEL, on by default; set 0 to disable).
        // Rejecting the round here and breaking was NOT sufficient for the
        // local-optimum claim: the selector commits a SET, and a set can have
        // DeltaF <= 0 purely from negative interaction between its members
        // (two moves that each pay off alone but whose anchors' ancestor
        // closures overlap, so the shared cost is charged once but the gains
        // do not add up) while individual actions in it still have
        // DeltaF > 0. Halting on the group verdict therefore could terminate
        // at a state that is not a local optimum of F.
        // Repair: on group rejection, fall back to verifying atomic actions
        // one at a time -- compute each candidate's TRUE DeltaF as a lone
        // commit (exact incremental DeltaQ minus lambda times its own
        // ancestor closure) and commit the best strictly-positive one. Only
        // when no single action improves F do we stop, which is exactly the
        // local-optimality condition. Each retained round still strictly
        // increases F, so termination is unaffected.
        int peeledMove = -1;
        double peeledDeltaF = 0.0;
        if(!accepted && !lamInf && PEEL){
          double bestDF = 1e-12;
          for(int ai=0; ai<(int)actions.size(); ai++){
            const auto& mv = moves[actionMove[ai]];
            // exact DeltaQ*2m/2 for moving this single vertex
            double ku=g.k[mv.vertex];
            double tf = ctotH.count(mv.from)?ctotH[mv.from]:0.0;
            double tt = ctotH.count(mv.to)?ctotH[mv.to]:0.0;
            double nullDelta = ((tf-ku)*(tf-ku) - tf*tf) + ((tt+ku)*(tt+ku) - tt*tt);
            double toC=0.0, toD=0.0;
            for(auto&[v,w]:g.adj[mv.vertex]){
              if((int)v==mv.vertex) continue;
              int cv = ((int)v<(int)leaf.size()) ? leaf[v] : -1;
              if(cv==mv.from) toC+=w;
              if(cv==mv.to)   toD+=w;
            }
            double gainOne = (2.0*(toD-toC))/2.0 - nullDelta/(2.0*g.m2);
            // its own ancestor closure cost
            std::vector<char> d(costTree.parent.size(),0);
            double costOne=0.0;
            for(int c: trueActions[ai].anchors)
              while(c>=0 && !d[c]){ d[c]=1; costOne+=costTree.weight[c]; c=costTree.parent[c]; }
            double dfOne = gainOne - lambda*costOne;
            if(dfOne > bestDF){ bestDF = dfOne; peeledMove = ai; peeledDeltaF = dfOne; }
          }
        }
        if(!accepted && peeledMove < 0) break;   // no set AND no single action improves F -> local optimum
        if(!accepted){
          // commit exactly the one peeled action this round
          for(int ai=0; ai<(int)chosen.selected_actions.size(); ai++)
            chosen.selected_actions[ai] = (ai==peeledMove) ? 1 : 0;
          nSelected = 1;
          candidate = leaf;
          candidate[moves[actionMove[peeledMove]].vertex] = moves[actionMove[peeledMove]].to;
          nPeel++;
          self<<k<<"\t"<<roundsRun<<"\tPEEL\t1\t0\t0\t0\t0\t"<<peeledDeltaF<<"\t0\t"
              <<peeledDeltaF<<"\t1\n"; self.flush();
        }
        anyRoundAccepted = true;

        // What this round actually asks the LLM to regenerate. Recomputed here
        // rather than reusing `trueDirty` above, because the peel repair can
        // replace the selected set after that was built.
        for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
          if(!chosen.selected_actions[ai]) continue;
          for(int c:trueActions[ai].anchors)
            while(c>=0 && c<(int)costTree.nodeKey.size()){
              auto key=costTree.nodeKey[c];
              if(!batchDirty.insert(key).second) break;   // ancestors already recorded
              c=costTree.parent[c];
            }
        }

        std::vector<int> movedV;
        Hierarchy Hc=committedHierarchy;
        for(int li=0;li<(int)Hc.levels.size();li++){
          int nextLabel=0;
          for(int c:Hc.levels[li]) nextLabel=std::max(nextLabel,c+1);
          while((int)Hc.levels[li].size()<g.n) Hc.levels[li].push_back(nextLabel++);
        }
        // BUGFIX (found via user's question): a round's next-round seed must be the
        // NEIGHBOURS of this round's movers, not the movers themselves -- a vertex that
        // just settled into its best community essentially never wants to move again
        // immediately, so seeding with only the movers makes the next round's queue
        // empty almost by construction, silently suppressing the cascading re-examination
        // that within-round propagation already does correctly (enqueue a move's
        // neighbours, not the mover). This mirrors exactly the in-round enqueue rule.
        for(int ai=0;ai<(int)chosen.selected_actions.size();ai++){
          if(!chosen.selected_actions[ai]) continue;
          const auto& mv=moves[actionMove[ai]];
          int repV=-1;
          if(TOPDOWN){ auto td=tdDest.find(mv.vertex); if(td!=tdDest.end()) repV=td->second; }
          if(repV<0){ auto rep=oldLeafRepresentative.find(mv.to);
                      if(rep==oldLeafRepresentative.end()) continue; repV=rep->second; }
          // (1)/(2) follow the partition: the prefilter needs the destination's
          // size next round, and the deferral needs to know that A is now
          // describing one entity it no longer contains.
          { auto itF=commSize.find(mv.from); if(itF!=commSize.end() && --itF->second<=0) commSize.erase(itF);
            commSize[mv.to]++; }
          { const auto& kc=kvComm[mv.vertex];
            auto ia=kc.find(mv.from), ib=kc.find(mv.to);
            double kvA=(ia!=kc.end())?ia->second:0.0, kvB=(ib!=kc.end())?ib->second:0.0;
            cinH[mv.from] -= 2.0*(kvA + g.self[mv.vertex]);
            cinH[mv.to]   += 2.0*(kvB + g.self[mv.vertex]); }
          if(DEFER && mv.vertex<(int)committedHierarchy.levels[0].size()
                   && repV<(int)committedHierarchy.levels[0].size()){
            int srcFine=committedHierarchy.levels[0][mv.vertex];
            int dstFine=committedHierarchy.levels[0][repV];
            if(srcFine!=dstFine && mv.from<1000000000){
              // An arrival has no prior summary, so nothing of it goes stale.
              staleCnt[srcFine]++; statDeferred++;
              // captured now because a community that loses its last member is
              // gone from the tree by flush time, and what has to be rewritten
              // then is its PARENT, whose child set changed
              if(committedHierarchy.levels.size()>1 &&
                 mv.vertex<(int)committedHierarchy.levels[1].size())
                staleParent.emplace(srcFine, committedHierarchy.levels[1][mv.vertex]);
              auto itS=fineSize.find(srcFine); if(itS!=fineSize.end() && --itS->second<=0) fineSize.erase(itS);
            }
            fineSize[dstFine]++;
          }
          movedV.push_back(mv.vertex);
          for(int li=0;li<(int)Hc.levels.size();li++)
            Hc.levels[li][mv.vertex]=committedHierarchy.levels[li][repV];
          // L1 (certificate scheme): commit this move's effect on the
          // PERSISTENT ctotH AND kvComm (the round's own `ctot`/`fresh`
          // were only tentative and are discarded for rejected actions --
          // only ACTUALLY selected moves may touch persistent state).
          // Propagate EXACT kvComm updates to every neighbour and
          // immediately recompute-and-rebuild-or-flag their certs (never
          // just drop-and-hope-for-requeue -- see
          // driver_margin_certificate.cxx's header comment on why that
          // was a confirmed real bug).
          int mvv=mv.vertex, mfrom=mv.from, mto=mv.to;
          double kmv=g.k[mvv];
          ctotH[mfrom] -= kmv;
          ctotH[mto]   += kmv;
          for(auto&[w,wt]:g.adj[mvv]){
            if((int)w==mvv) continue;
            kvComm[w][mfrom]-=wt; if(kvComm[w][mfrom]<=1e-12) kvComm[w].erase(mfrom);
            kvComm[w][mto]+=wt;
            int Aw=leaf[(int)w];
            recheckAllH((int)w);   // k_{w,A} may have shifted (mfrom/mto == Aw case)
            if(mfrom!=Aw){ double H=computeHc((int)w,Aw,mfrom);
              if(H>=-1e-12) gateLevel2((int)w,mfrom,H/g.m2); else buildCertH((int)w,Aw,mfrom,H); }
            if(mto!=Aw){ double H=computeHc((int)w,Aw,mto);
              if(H>=-1e-12) gateLevel2((int)w,mto,H/g.m2); else buildCertH((int)w,Aw,mto,H); }
          }
          leaf[mvv]=mto;
          { auto lst=certsOfVertex[mvv]; for(int cid:lst) dropCertH(mvv, certStore[cid].D); }
          int A2=leaf[mvv];
          for(auto&[D,w]:kvComm[mvv]){
            if(D==A2) continue;
            double H=computeHc(mvv,A2,D);
            if(H>=-1e-12) gateLevel2(mvv,D,H/g.m2); else buildCertH(mvv,A2,D,H);
          }
          sweepSigH(mto);
          sweepSigDH(mfrom);
        }
        leaf = TOPDOWN ? Hc.levels.back() : Hc.levels[0];
        // Hc.Q intentionally NOT recomputed here. It is a whole-graph O(|E|)
        // modularity call per accepted round -- profiled at 99.8-100.0% of the
        // commit phase and 51-57% of total cost-aware time -- and nothing in
        // this driver ever reads it: the commit gate uses the incremental
        // actualGain, and the metrics line calls modularity() separately,
        // outside the timed region. (detect_iter maintains its own H.Q for its
        // own convergence test; that is unaffected.)
        committedHierarchy=Hc;
        if(INC_TREE && costTreeValid){
          HierCostTree& T=costTreeCache;
          bool ok=!T.ids.empty();
          if(ok && (int)T.vertexLeaf.size()<g.n) T.vertexLeaf.resize(g.n,-1);
          // Vertices the tree does not cover yet. A batch pads the hierarchy
          // with a fresh label at every level for each new vertex, so a full
          // rebuild would create a whole chain of one-member nodes for it;
          // the patch has to do the same or it drifts from the rebuild by
          // exactly those nodes (measured: 492 missing by the end of
          // Cit-HepTh before this was added). A vertex already in range can
          // also become eligible by gaining its first edge, so batch
          // endpoints are considered too -- both sets are O(batch), not O(N).
          if(ok){
            std::vector<int> fresh;
            for(int v : pendingTreeV) if(v<g.n && g.k[v]>0 && T.vertexLeaf[v]<0) fresh.push_back(v);
            std::sort(fresh.begin(),fresh.end());
            fresh.erase(std::unique(fresh.begin(),fresh.end()),fresh.end());
            const int nlev=(int)T.ids.size();
            for(int v : fresh){
              if(T.vertexLeaf[v]>=0) continue;
              int child=-1;
              for(int li=0; li<nlev; li++){
                if(v>=(int)Hc.levels[li].size()) break;
                const int c=Hc.levels[li][v];
                auto it=T.ids[li].find(c);
                int node;
                if(it!=T.ids[li].end()){
                  node=it->second;
                  T.weight[node]+=1.0;                 // one more member (li=0) or child
                } else {
                  node=(int)T.parent.size();
                  T.ids[li][c]=node; T.parent.push_back(-1);
                  T.weight.push_back(1.0); T.nodeKey.push_back({li,c});
                }
                nodeWeightPersist[bkey(li,c)]=T.weight[node];
                if(li==0) T.vertexLeaf[v]=node; else T.parent[child]=node;
                child=node;
                if(it!=T.ids[li].end()) break;          // ancestors already exist and already count this subtree
              }
            }
          }
          pendingTreeV.clear();
          for(int v : movedV){
            if(!ok) break;
            const int newComm0 = Hc.levels[0][v];
            auto it = T.ids[0].find(newComm0);
            if(it==T.ids[0].end()){ ok=false; break; }      // destination not in the tree -> fall back
            const int newNode=it->second, oldNode=T.vertexLeaf[v];
            if(newNode==oldNode) continue;
            T.weight[newNode]+=1.0;
            nodeWeightPersist[bkey(0,newComm0)]=T.weight[newNode];
            T.vertexLeaf[v]=newNode;
            // repOfComm was refreshed by the index block only when the tree was
            // rebuilt. Keeping the tree valid skips that refresh, and the moves
            // are exactly what invalidates it -- a stale representative makes a
            // later mover copy the wrong level path, which changed the
            // partition (Q 0.83532 against 0.83549 on Facebook).
            repOfComm[newComm0]=v;
            if(oldNode<0) continue;                          // an arrival: it was in no leaf before
            T.weight[oldNode]-=1.0;
            nodeWeightPersist[bkey(0,T.nodeKey[oldNode].second)]=T.weight[oldNode];
            if(T.weight[oldNode]>1e-9) continue;
            // the leaf emptied: it leaves the tree and each ancestor loses a child
            T.ids[0].erase(T.nodeKey[oldNode].second);
            nodeWeightPersist.erase(bkey(0,T.nodeKey[oldNode].second));
            repOfComm.erase(T.nodeKey[oldNode].second);
            int par=T.parent[oldNode]; T.parent[oldNode]=-1;
            while(par>=0){
              T.weight[par]-=1.0;
              const auto key=T.nodeKey[par];
              if(T.weight[par]>1e-9){ nodeWeightPersist[bkey(key.first,key.second)]=T.weight[par]; break; }
              T.ids[key.first].erase(key.second);
              nodeWeightPersist.erase(bkey(key.first,key.second));
              const int pp=T.parent[par]; T.parent[par]=-1; par=pp;
            }
          }
          if(!ok) costTreeValid=false;
          if(VALIDATE_TREE && costTreeValid){
            HierCostTree F=make_cost_tree(g,committedHierarchy);
            double dw=0.0; long miss=0;
            for(int li=0; li<(int)F.ids.size(); li++)
              for(auto&[c,id] : F.ids[li]){
                auto i2=T.ids[li].find(c);
                if(i2==T.ids[li].end()){ miss++; continue; }
                dw=std::max(dw,std::fabs(F.weight[id]-T.weight[i2->second]));
              }
            fprintf(stderr,"[validate-tree] batch=%d round=%d max|dw|=%.3g missing=%ld\n",k,roundsRun,dw,miss);
          }
        } else costTreeValid=false;
        { long nd=0, nt=(long)costTree.parent.size();
          for(char d:chosen.dirty_nodes) nd+=d;
          dpf<<k<<"\t"<<roundsRun<<"\t"<<nd<<"\t"<<nt<<"\t"<<(nt?(double)nd/nt:0.0)<<"\n"; dpf.flush(); }
        // next round re-examines whatever the certificate system flagged
        // while committing this round's moves ("consume" queueH into
        // roundSeed, then clear it so it can accumulate fresh next round).
        roundSeed.assign(queueH.begin(), queueH.end());
        for(int v: queueH) queueEpoch[v]=-1;
        queueH.clear();
      }
      if(!anyRoundAccepted){
        dpf<<k<<"\t0\t0\t0\t0.0\n"; dpf.flush();
      }

      Hierarchy Hc = committedHierarchy;
      // L3: gap-conditional periodic audit (Theorem competitive), matching
      // the fast path's own policy and the paper's actual ski-rental design
      // -- was previously an UNCONDITIONAL full rebuild every K batches
      // regardless of any measured drift, which (per the per-batch latency
      // breakdown) dominates hier-mode's total time (rebuild batches cost
      // ~20-30x a regular batch). Every K batches, cheaply PROBE the drift
      // via a single local_moving pass (not a full detect_iter) and only
      // pay for the expensive full rebuild if the measured gap reaches R.
      // ADDED: a second, structural trigger (DYLO_SIGMA). The modularity-gap
      // test alone has a blind spot that is not hypothetical -- it was
      // measured this session. A newly-arrived vertex is merged into
      // community D only if gain - lambda*closure(D) > 0, and closure(D) is
      // ~|D|, so at lambda=1e-1 no community larger than ~gain/lambda ~ 10
      // members ever accepts an arrival: every new vertex is parked in its
      // own singleton community, permanently. That degrades the index (up
      // to 59% of Cit-HepPh's communities became one-member) while moving
      // modularity by ~5e-4, because the parked vertices are a negligible
      // FRACTION OF NODES (0.27% on Facebook) even though they are a large
      // fraction of COMMUNITIES. So the drift monitor cannot see the very
      // degradation the rebuild exists to repair: measured audits=1/4 with
      // rebuilds=0 on Facebook/Cit-HepTh while singletons reached 37.9%/43.4%.
      // Rebuilding when the singleton fraction crosses sigma closes that
      // gap. It touches only the L3 trigger -- Q's definition, the cost
      // model, and the min-cut solver are all unchanged.
      if((R<1e17 || SIGMA_ONLY) && (++batchesSinceRebuild>=auditK || (FINAL_AUDIT && k==nB))){
        std::vector<int> full;
        double gap=0.0;
        batchesSinceRebuild=0; nAudit++;
        if(!SIGMA_ONLY){
          full=leaf; local_moving(g,full,rng); renumber(full);
          gap=modularity(g,full)-modularity(g,leaf);
          driftAcc += std::max(0.0, gap-gapAtLastRebuild);   // ski-rental: pay rent
        }
        double sfrac=0.0;
        // commSize is already maintained incrementally for the size prefilter,
        // and is exactly the per-community count this needed, so the O(N) pass
        // that rebuilt it is gone. What remains is O(#communities), which is
        // three orders of magnitude smaller on the large graphs (DBLP: 760
        // communities against 317K vertices).
        if(SIGMA<1.0){
          int single=0, live=0;
          for(auto&[c,sz]:commSize){ if(sz<=0) continue; live++; if(sz==1) single++; }
          sfrac = live? (double)single/(double)live : 0.0;
          if(std::getenv("DYLO_VALIDATE_SFRAC")){
            std::unordered_map<int,int> csz2;
            for(int u=0;u<g.n;u++) if(g.k[u]>0) csz2[leaf[u]]++;
            int s2=0; for(auto&[c,x]:csz2) if(x==1) s2++;
            double f2 = csz2.empty()?0.0:(double)s2/(double)csz2.size();
            fprintf(stderr,"[validate-sfrac] batch=%d commSize: live=%d single=%d f=%.4f | scan: live=%zu single=%d f=%.4f\n",
                    k,live,single,sfrac,csz2.size(),s2,f2);
          }
        }
        // TARGETED SINGLETON RECLAMATION (DYLO_RECLAIM=1). A full rebuild
        // costs ~0.99 (it renumbers everything, dirtying the whole tree)
        // whereas ordinary incremental batches cost ~1e-4 -- measured this
        // session. Paying ~0.99 merely to re-home a few dozen parked
        // vertices is enormously out of proportion. Instead, when the
        // structural trigger fires, re-home ONLY the singletons: each is
        // moved to its best-gain neighbouring community directly, dirtying
        // just that community (and its ancestors) rather than the entire
        // partition. lambda is deliberately NOT consulted here: the whole
        // reason these vertices are stuck is that gain - lambda*|D| < 0 for
        // every community bigger than ~gain/lambda, and that test omits the
        // benefit actually being bought -- eliminating a redundant one-entity
        // summary from the index. Reclamation is gated on the same sigma
        // trigger, so it only runs when the index has measurably degraded.
        if(RECLAIM && SIGMA<1.0 && sfrac>=SIGMA && (SIGMA_ONLY || driftAcc<R)){
          // The one remaining O(N) pass, and it runs only when the trigger has
          // actually fired -- a singleton community's member cannot be named
          // from the size table alone. ctotRe was a third full pass rebuilding
          // what ctotH already holds.
          std::vector<int> singles;
          {
            // O(|Pi|) instead of O(N): a singleton community's sole member IS the
            // representative the repair path already maintains, so the size table
            // and repOfComm together name it without touching the vertex array.
            for(auto&[c,sz] : commSize){
              if(sz!=1) continue;
              auto r=repOfComm.find(c);
              if(r==repOfComm.end()) { statRepMiss++; continue; }
              int u=r->second;
              if(u>=0 && u<g.n && g.k[u]>0 && leaf[u]==c) singles.push_back(u);
              else statRepStale++;
            }
            std::sort(singles.begin(), singles.end());
            if(VALIDATE_AUDIT){                       // equivalence check vs the O(N) scan
              std::vector<int> ref;
              for(int u=0;u<g.n;u++) if(g.k[u]>0){ auto it=commSize.find(leaf[u]);
                 if(it!=commSize.end() && it->second==1) ref.push_back(u); }
              std::sort(ref.begin(), ref.end());
              if(ref!=singles){ statSinglesMismatch++;
                fprintf(stderr,"[validate-audit] batch=%d fast=%zu ref=%zu\n",k,singles.size(),ref.size()); }
              else statSinglesOK++;
            }
          }
          std::unordered_map<int,double>& ctotRe = ctotH;
          int moved=0;
          std::vector<std::pair<int,std::pair<int,int>>> movedFrom;
          std::unordered_map<int,int> reclaimRep;
          for(int u: singles){
            int ci=leaf[u]; double ku=g.k[u];
            std::unordered_map<int,double> kin;
            for(auto&[v,w]:g.adj[u]) if((int)v!=u) kin[leaf[v]]+=w;
            int best=ci; double bestgain=0.0;   // staying alone: zero internal edges
            for(auto&[c,kic]:kin){
              if(c==ci) continue;
              double gn=kic - ctotRe[c]*ku/g.m2;
              if(gn>bestgain+1e-12){ bestgain=gn; best=c; }
            }
            if(best!=ci){
              // strongest neighbour already inside `best` decides which FINE
              // leaf the reclaimed vertex joins, mirroring the move path. Under
              // TOPDOWN the destination id names a level-back() community, so
              // the whole level path has to come from a real member.
              double bw=-1.0; int bu=-1;
              for(auto&[v2,w2]:g.adj[u]){
                if((int)v2==u) continue;
                if((int)v2<(int)leaf.size() && leaf[(int)v2]==best && w2>bw){ bw=w2; bu=(int)v2; }
              }
              reclaimRep[u]=bu;
              ctotRe[ci]-=ku; ctotRe[best]+=ku;
              movedFrom.push_back({u,{ci,best}});
              leaf[u]=best; moved++;
            }
          }
          if(moved>0){
            nReclaim++; nReclaimed+=moved;
            if(FAST_RECLAIM){
              // ids stay put, so only what actually changed is repaired.
              int nlev=(int)committedHierarchy.levels.size();
              // repOfComm already IS "community -> a member vertex", maintained by
              // the insertion and move paths; reclamation only ADDS members to a
              // destination, so its entry stays a valid member. Rebuilding the same
              // map with an O(N) scan and a hash insert per vertex was redundant.
              const std::unordered_map<int,int>& rep = repOfComm;
              if(VALIDATE_AUDIT){
                std::unordered_map<int,int> ref;
                for(int v=0;v<g.n;v++) if(g.k[v]>0) ref.emplace(leaf[v],v);
                for(auto&[u2,AB2] : movedFrom){
                  int B2=AB2.second;
                  auto a=rep.find(B2); auto b=ref.find(B2);
                  bool oka = (a!=rep.end() && a->second<g.n && leaf[a->second]==B2);
                  bool okb = (b!=ref.end());
                  if(oka!=okb) statRepDiff++; else statRepSame++;
                }
              }
              for(auto&[u,AB] : movedFrom){
                int A=AB.first, B=AB.second; double ku=g.k[u];
                // captured BEFORE the level path below is overwritten
                const int srcFine0 = (nlev>0 && u<(int)committedHierarchy.levels[0].size())
                                   ? committedHierarchy.levels[0][u] : -1;
                ctotH[A]-=ku; ctotH[B]+=ku;
                for(auto&[v,w]:g.adj[u]){
                  if((int)v==u) continue;
                  auto& kc=kvComm[(int)v];
                  double& fa=kc[A]; fa-=w; if(fa<1e-12 && fa>-1e-12) kc.erase(A);
                  kc[B]+=w;
                }
                int rv=-1;
                { auto rr=reclaimRep.find(u); if(rr!=reclaimRep.end()) rv=rr->second; }
                if(rv<0){ auto rit=rep.find(B); if(rit!=rep.end()) rv=rit->second; }
                // The braces matter: without them the `else` binds to the inner
                // bounds check, so exactly when u is OUT of range the code took
                // the branch that writes levels[nlev-1][u] with no check at all.
                // u is out of range precisely for a vertex that arrived after
                // this hierarchy was built, i.e. on any stream where entities
                // appear over time.
                if(rv>=0 && rv!=u){
                  for(int li=0; li<nlev; li++)
                    if(u<(int)committedHierarchy.levels[li].size() &&
                       rv<(int)committedHierarchy.levels[li].size())
                      committedHierarchy.levels[li][u]=committedHierarchy.levels[li][rv];
                } else if(nlev>0 && u<(int)committedHierarchy.levels[nlev-1].size()){
                  committedHierarchy.levels[nlev-1][u]=B;
                }
                // Reclamation moves vertices too, so it re-summarizes exactly
                // like a committed round does: the destination is dirtied now,
                // and under deferral the source only accrues a residual. Fast
                // reclamation keeps ids stable, so the counters stay valid and
                // there is nothing to clear -- but the cost is real and was
                // being recorded nowhere.
                { auto itF=commSize.find(A); if(itF!=commSize.end() && --itF->second<=0) commSize.erase(itF);
                  commSize[B]++; }
                { const auto& kc=kvComm[u];
                  auto ia=kc.find(A), ib=kc.find(B);
                  double kvA=(ia!=kc.end())?ia->second:0.0, kvB=(ib!=kc.end())?ib->second:0.0;
                  cinH[A] -= 2.0*(kvA + g.self[u]);
                  cinH[B] += 2.0*(kvB + g.self[u]); }
                if(nlev>0 && u<(int)committedHierarchy.levels[0].size()){
                  const int dstFine=committedHierarchy.levels[0][u];
                  reclaimDirtyFine.push_back(dstFine);
                  if(DEFER && srcFine0>=0 && srcFine0!=dstFine){
                    fineSize[dstFine]++;
                    staleCnt[srcFine0]++; statDeferred++;
                    if(committedHierarchy.levels.size()>1 &&
                       u<(int)committedHierarchy.levels[1].size())
                      staleParent.emplace(srcFine0, committedHierarchy.levels[1][u]);
                    auto itS=fineSize.find(srcFine0);
                    if(itS!=fineSize.end() && --itS->second<=0) fineSize.erase(itS);
                  } else if(DEFER && srcFine0!=dstFine) fineSize[dstFine]++;
                  if(!DEFER && srcFine0>=0 && srcFine0!=dstFine) reclaimDirtyFine.push_back(srcFine0);
                }
              }
              Hc=committedHierarchy; costTreeValid=false;
              std::fill(marginH.begin(), marginH.end(), -1.0);
              if(VALIDATE_RECLAIM){
                std::unordered_map<int,double> ct2;
                for(int u2=0;u2<g.n;u2++) ct2[leaf[u2]] += g.k[u2];
                std::vector<std::unordered_map<int,double>> kv2(g.n);
                for(int v=0;v<g.n;v++) for(auto&[u2,w]:g.adj[v]) kv2[v][leaf[(int)u2]]+=w;
                double dct=0, dkv=0;
                for(auto&[c,x]:ct2) dct=std::max(dct,std::fabs(x-ctotH[c]));
                for(int v=0;v<g.n;v++){
                  for(auto&[c,x]:kv2[v]){
                    auto it2=kvComm[v].find(c);
                    double have=(it2!=kvComm[v].end())?it2->second:0.0;
                    dkv=std::max(dkv,std::fabs(x-have));
                  }
                  // and the other direction: entries we hold that truth does not
                  for(auto&[c,have]:kvComm[v]){
                    auto it2=kv2[v].find(c);
                    double want=(it2!=kv2[v].end())?it2->second:0.0;
                    dkv=std::max(dkv,std::fabs(want-have));
                  }
                }
                fprintf(stderr,"[validate-reclaim] batch=%d moved=%d  max|dctot|=%.3e  max|dkv|=%.3e\n",
                        k,moved,dct,dkv);
              }
            } else {
            Hc = COMBINED ? build_levels_combined(g,leaf,inst,rng,TD_MAXSZ,HIER_GAMMA)
                          : build_levels_from_leaf(g,leaf,inst,rng,HIER_GAMMA);
            leaf=Hc.levels[0];
            // Renumbering invalidates every community-keyed counter: sizes,
            // fine-community sizes and residuals are all keyed by ids that no
            // longer mean the same thing. A rebuild also regenerates every
            // summary, so clearing the residuals is not merely safe -- it is
            // what actually happened, and the batch is charged the whole tree.
            commSize.clear(); fineSize.clear(); staleCnt.clear(); staleParent.clear(); pendingTreeV.clear(); cinH.clear();
            qAtRefresh.clear(); batchAllDirty=true;
            committedHierarchy=Hc; costTreeValid=false;
            ctotH.clear();
            for(int u=0;u<g.n;u++) ctotH[leaf[u]] += g.k[u];
            std::fill(marginH.begin(), marginH.end(), -1.0);
            kvComm.assign(g.n, {});
            for(int v=0;v<g.n;v++) for(auto&[u,w]:g.adj[v]) kvComm[v][leaf[u]]+=w;
            }
            // Reclamation used to clear every certificate and both drift heaps
            // and rebuild from scratch, walking kvComm for EVERY vertex. That
            // is O(|E|) per reclamation -- profiled at 968 of DBLP's 1633 ms,
            // 59% of the whole run, across 6.47M kvComm entries -- and it is
            // unnecessary twice over. Only 4839 of 317K vertices moved, so only
            // their own certificates and their neighbours' inputs changed; and
            // the community-degree drift that reaches everyone else is exactly
            // what sigHeapH/sigDHeapH exist to catch, so clearing the heaps
            // destroyed the incremental machinery in order to replace it.
            //
            // The repair is the one the round-commit path already performs for
            // an ordinary move: fix up each mover's own certificates, re-check
            // its neighbours, and sweep the two touched communities' heaps.
            if(LAZY_RECLAIM){
              for(auto&[u,AB] : movedFrom){
                const int A2=leaf[u], mfrom=AB.first, mto=AB.second;
                { auto lst=certsOfVertex[u]; for(int cid:lst) dropCertH(u, certStore[cid].D); }
                for(auto&[D,w]:kvComm[u]){
                  if(D==A2) continue;
                  double H=computeHc(u,A2,D);
                  if(H>=-1e-12) gateLevel2(u,D,H/g.m2); else buildCertH(u,A2,D,H);
                }
                for(auto&[w,wt]:g.adj[u]){
                  if((int)w==u) continue;
                  recheckAllH((int)w);
                }
                sweepSigH(mto); sweepSigDH(mfrom);
              }
            } else {
              certStore.clear(); vD2cert.clear(); certsOfVertex.assign(g.n,{});
              m2HeapH=decltype(m2HeapH)(); sigHeapH.clear(); sigDHeapH.clear();
              for(int v=0;v<g.n;v++){
                int A=leaf[v];
                for(auto&[D,w]:kvComm[v]){
                  if(D==A) continue;
                  double H=computeHc(v,A,D);
                  if(H<-1e-12) buildCertH(v,A,D,H);
                }
              }
            }
          }
          sfrac=0.0;   // handled by reclamation; do not also fire a full rebuild
        }
        if(!SIGMA_ONLY && (driftAcc>=R || sfrac>=SIGMA)){
          gapAtLastRebuild=gap; driftAcc=0.0;
          Hierarchy Hr=detect_iter(g,inst,42);
          leaf=Hr.levels[0];
          Hc = COMBINED ? build_levels_combined(g,leaf,inst,rng,TD_MAXSZ,HIER_GAMMA)
                          : build_levels_from_leaf(g,leaf,inst,rng,HIER_GAMMA);
          leaf=Hc.levels[0];
          commSize.clear(); fineSize.clear(); staleCnt.clear(); staleParent.clear(); pendingTreeV.clear(); cinH.clear();
            qAtRefresh.clear(); batchAllDirty=true;   // see the reclamation site
          nRebuild++; rbf<<k<<"\n"; rbf.flush();
          committedHierarchy=Hc; costTreeValid=false;
          // L1 state must be re-derived after a full rebuild: community ids
          // are freshly renumbered and prior margins/certificates no
          // longer apply (kvComm keys are stale community ids).
          ctotH.clear();
          for(int u=0;u<g.n;u++) ctotH[leaf[u]] += g.k[u];
          std::fill(marginH.begin(), marginH.end(), -1.0);
          kvComm.assign(g.n, {});
          for(int v=0;v<g.n;v++) for(auto&[u,w]:g.adj[v]) kvComm[v][leaf[u]]+=w;
          certStore.clear(); vD2cert.clear(); certsOfVertex.assign(g.n,{});
          m2HeapH=decltype(m2HeapH)(); sigHeapH.clear(); sigDHeapH.clear();
          for(int v=0;v<g.n;v++){
            int A=leaf[v];
            for(auto&[D,w]:kvComm[v]){
              if(D==A) continue;
              double H=computeHc(v,A,D);
              if(H<-1e-12) buildCertH(v,A,D,H);
            }
          }
        }
      }
      // (2) THETA FLUSH. A deferred source keeps listing entities that have
      // left. Once those residuals reach THETA of what its summary claims to
      // describe, the community is regenerated and its residuals clear. This
      // is where the deferral's saving is actually realised: ell departures
      // cost one regeneration instead of ell. Inside the timed region, since
      // deciding it is per-batch work the pipeline really does.
      if(!reclaimDirtyFine.empty() && !batchAllDirty){
        if(!costTreeValid){ costTreeCache=make_cost_tree(g,committedHierarchy); costTreeValid=true; }
        const HierCostTree& ct=costTreeCache;
        if(!ct.ids.empty())
          for(int c:reclaimDirtyFine){
            auto idIt=ct.ids[0].find(c);
            if(idIt==ct.ids[0].end()) continue;
            for(int n=idIt->second; n>=0 && n<(int)ct.nodeKey.size(); n=ct.parent[n])
              if(!batchDirty.insert(ct.nodeKey[n]).second) break;
          }
      }
      if(DEFER && !staleCnt.empty()){
        if(!costTreeValid){ costTreeCache=make_cost_tree(g,committedHierarchy); costTreeValid=true; }
        const HierCostTree& ct=costTreeCache;
        if(!ct.ids.empty()){
          // Members of the stale communities only -- one pass over the vertices
          // rather than a full community index, since the stale set is small.
          auto chargeFrom=[&](int lvl,int comm){
            if(lvl>=(int)ct.ids.size()) return;
            auto idIt=ct.ids[lvl].find(comm);
            if(idIt==ct.ids[lvl].end()) return;
            for(int n=idIt->second; n>=0 && n<(int)ct.nodeKey.size(); n=ct.parent[n])
              if(!batchDirty.insert(ct.nodeKey[n]).second) break;
          };
          for(auto it=staleCnt.begin(); it!=staleCnt.end(); ){
            const int c=it->first, st=it->second;
            auto fs=fineSize.find(c);
            const int live=(fs==fineSize.end())?0:fs->second;

            // A community that lost its last member is not regenerated -- its
            // summary is DELETED. What changes is its PARENT, whose child set
            // shrank, so the parent's closure is what this costs. Charging the
            // vanished community's own weight (what the fraction rule did)
            // prices a rewrite that never happens, and it fired on 97-99.8% of
            // all flushes -- which is why no threshold could ever amortise.
            if(live==0){
              auto pit=staleParent.find(c);
              if(pit!=staleParent.end()) chargeFrom(1,pit->second);
              statFlushed++; statFlushVanished++;
              staleParent.erase(c); qAtRefresh.erase(c);
              it=staleCnt.erase(it);
              continue;
            }

            bool fire; double qNow=0.0;
            if(DEFER_QUALITY){
              // q_A = in_A/2m - (sigma_A/2m)^2, both terms maintained
              // incrementally, so the test is O(1) per stale community rather
              // than a pass over the vertices and their adjacency.
              auto iI=cinH.find(c), iT=ctotH.find(c);
              const double inA=(iI!=cinH.end())?iI->second:0.0;
              const double sigA=(iT!=ctotH.end())?iT->second:0.0;
              qNow = (g.m2>0.0) ? (inA/g.m2 - (sigA/g.m2)*(sigA/g.m2)) : 0.0;
              if(VALIDATE_QC){
                // compare against `leaf`, which is what ctotH/cinH are keyed by;
                // levels[0] lags it for arrivals not yet folded into the hierarchy
                double i2=0.0,s2=0.0;
                for(int v=0;v<g.n && v<(int)leaf.size();v++){
                  if(g.k[v]<=0 || leaf[v]!=c) continue;
                  s2+=g.k[v]; i2+=2.0*g.self[v];
                  for(auto&[u,w]:g.adj[v]) if((int)u<(int)leaf.size() && leaf[(int)u]==c) i2+=w;
                }
                fprintf(stderr,"[validate-qc] c=%d in:%.3f/%.3f sig:%.3f/%.3f\n",c,inA,i2,sigA,s2);
              }
              auto qit=qAtRefresh.find(c);
              if(qit==qAtRefresh.end()){ qAtRefresh[c]=qNow; fire=false; }
              else {
                const double base=std::max(std::fabs(qit->second),1e-12);
                fire = std::fabs(qNow-qit->second)/base >= DEFER_EPS;
              }
            } else {
              fire = (live+st>0) && (double)st/(double)(live+st) >= THETA;
            }

            if(fire){
              chargeFrom(0,c);
              statFlushed++; statFlushQuality++;
              qAtRefresh[c]=qNow;            // the rewrite describes the CURRENT structure
              staleParent.erase(c);
              it=staleCnt.erase(it);
            } else ++it;
          }
        }
      }
      auto t1=std::chrono::high_resolution_clock::now();
      double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
      // The batch's REALISED re-summarization cost, taken from the schedule the
      // driver actually followed. An offline scorer reading com_k*.txt cannot
      // see a deferral: the partition is byte-identical whether A is refreshed
      // now or in ten batches, so the saving is only visible from here.
      {
        if(!costTreeValid){ costTreeCache=make_cost_tree(g,committedHierarchy); costTreeValid=true; }
        const HierCostTree& ct=costTreeCache;
        double tot=0.0; for(double w:ct.weight) tot+=w;
        double cst=0.0;
        if(batchAllDirty) cst=tot;
        else for(const auto& key:batchDirty)
          if(key.first<(int)ct.ids.size()){
            auto i2=ct.ids[key.first].find(key.second);
            if(i2!=ct.ids[key.first].end()) cst+=ct.weight[i2->second];
          }
        resum<<k<<"\t"<<(batchAllDirty?1:0)<<"\t"<<batchDirty.size()<<"\t"
             <<cst<<"\t"<<tot<<"\t"<<(tot>0.0?cst/tot:0.0)<<"\n"; resum.flush();
      }
      const auto& repK = TOPDOWN ? Hc.levels.back() : Hc.levels[0];
      const bool lastB = (k==nB) || (STREAM_BUDGET>0.0 &&
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-streamT0).count()>=STREAM_BUDGET);
      const double qOut = (Q_AT_END && !lastB) ? -1.0 : modularity(g,repK);
      const int   nOut = (Q_AT_END && !lastB) ? -1   : count_comm(repK);
      metrics<<k<<"\t"<<ms<<"\t"<<qOut<<"\t"<<nOut
             <<"\t"<<roundsRun<<"\n"; metrics.flush();
      if(!noCom){ write_levels(k, Hc); write_com(k, repK); }
      continue;
    }

    // ---- FAST optimized affected-region update (Alg 1 + MARGIN LEMMA gate) ----
    // New vertices -> their own fresh dense community; grow structures.
    if((int)C.size()<g.n){
      for(int v=(int)C.size(); v<g.n; v++){ C.push_back(nextComm++); ctotF.push_back(0.0); }
      inA.assign(g.n, 0); margin.resize(g.n, -1.0); pert.resize(g.n, 0.0);
    }
    if((int)kbuf.size()<nextComm) kbuf.resize(nextComm, 0.0);
    // Incremental community-degree update + MARGIN-LEMMA gate. Insertion-only stream:
    // a vertex can only leave its community if its accrued perturbation (new incident weight
    // to OTHER communities, plus propagation from moved neighbours) exceeds its gain margin.
    // A well-attached hub (large margin) that gains a few edges cannot flip -> skip its O(deg)
    // histogram entirely. Residual drift (global degree growth) is caught by the periodic audit.
    std::vector<int> A; std::vector<int> resetA; std::vector<int> resetP;
    auto bumpPert=[&](uint32_t x,double w){ if(pert[x]==0.0) resetP.push_back((int)x); pert[x]+=w; };
    auto enq=[&](uint32_t x){ if(!inA[x] && pert[x] > margin[x]){ inA[x]=1; A.push_back((int)x); resetA.push_back((int)x); } };
    for(auto&[u,v]:ins){
      if(u==v){ ctotF[C[u]] += 2.0; continue; }
      ctotF[C[u]] += 1.0; ctotF[C[v]] += 1.0;             // matches g.k[u]+=1, g.k[v]+=1
      if(C[u]!=C[v]){ bumpPert(u,1.0); bumpPert(v,1.0); } // cross-community insertion perturbs both
    }
    for(int x:resetP) enq((uint32_t)x);                   // gate: only vertices whose pert exceeds margin
    size_t head=0;
    std::vector<int> touched; touched.reserve(256);        // hoisted scratch (reused per vertex)
    while(head<A.size()){
      int u=A[head++]; int ci=C[u]; double ku=g.k[u];
      // neighbour-community histogram via scratch array (O(deg), reset only touched entries)
      touched.clear();
      for(auto&[w,wt]:g.fadj[u]){ int cw=C[w]; if(kbuf[cw]==0.0) touched.push_back(cw); kbuf[cw]+=wt; }
      ctotF[ci]-=ku;
      double stay = kbuf[ci] - ctotF[ci]*ku/g.m2;
      int best=ci; double bestgain=stay, second=-1e300;
      for(int c:touched){ if(c==ci) continue; double gn=kbuf[c]-ctotF[c]*ku/g.m2;
        if(gn>bestgain+1e-12){ second=bestgain; bestgain=gn; best=c; }
        else if(gn>second) second=gn; }
      ctotF[best]+=ku;
      // margin = slack of the chosen community over the best rival: incident weight u would
      // need to gain elsewhere to flip. No rival community in reach -> effectively unflippable.
      margin[u] = (second > -1e299) ? (bestgain - second) : 1e18;
      if(margin[u] < 0) margin[u]=0;
      pert[u]=0.0;
      if(best!=ci){ C[u]=best;
        // propagate: a neighbour's landscape shifted by up to w(u,*); re-gate it
        for(auto&[w,wt]:g.fadj[u]){ bumpPert(w,wt); enq(w); } }
      for(int c:touched) kbuf[c]=0.0;                      // reset scratch
    }
    if(std::getenv("DYLO_DEBUG")){ double vol=0; for(int x:resetA){ vol+=g.k[x]; }
      fprintf(stderr,"  b%d |A|=%zu vol(A)=%.0f vol/|E|=%.4f gated_in=%zu\n",k,resetA.size(),vol,vol/g.m2,resetP.size()); }
    for(int x:resetA) inA[x]=0;                            // clear affected flags for next batch
    for(int x:resetP) pert[x]=0.0;                         // clear perturbations not consumed
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    metrics<<k<<"\t"<<ms<<"\t"<<modularity_fadj(g,C)<<"\t"<<count_comm(C)<<"\t0\n"; metrics.flush();
    if(!noCom) write_com(k,C);
    if(R<1e17 && ++batchesSinceRebuild>=auditK){   // skip audit entirely in never-rebuild config
      g.rebuild_adj();                              // untimed: audit/rebuild use adj-based ops
      std::vector<int> full=C; local_moving(g,full,rng); renumber(full);
      double gap=modularity(g,full)-modularity(g,C);
      batchesSinceRebuild=0; nAudit++;
      driftAcc += std::max(0.0, gap-gapAtLastRebuild);   // ski-rental: pay rent
      if(driftAcc>=R){ gapAtLastRebuild=gap; driftAcc=0.0;
        nRebuild++; rbf<<k<<"\n"; rbf.flush(); Hierarchy H2=detect_iter(g,inst,42); C=H2.levels.back();
        // rebuild incremental state after a from-scratch rebuild (dense renumbered ids)
        nextComm=0; for(int v=0;v<g.n;v++) nextComm=std::max(nextComm,C[v]+1);
        ctotF.assign(nextComm,0.0); for(int v=0;v<g.n;v++) ctotF[C[v]]+=g.k[v];
        kbuf.assign(nextComm,0.0); }
    }
  }
  fprintf(stderr,"[dylo:%s] done %d batches; audits=%d rebuilds=%d bucket_adds=%ld bucket_flushes=%ld bucket_fallback=%ld\n",
          algo.c_str(),nB,nAudit,nRebuild,statBucketAdd,statBucketFlush,statBucketFallback);
  fprintf(stderr,"[dylo:%s] total_edges=%ld vertex_triggers=%ld edges_per_trigger=%.2f\n",
          algo.c_str(),statTotalEdges,statVertexTriggers,
          statVertexTriggers>0?(double)statTotalEdges/statVertexTriggers:0.0);
  fprintf(stderr,"[audit] singles_ok=%ld mismatch=%ld rep_miss=%ld rep_stale=%ld rep_same=%ld rep_diff=%ld\n",statSinglesOK,statSinglesMismatch,statRepMiss,statRepStale,statRepSame,statRepDiff);
  fprintf(stderr,"[dylo:%s] reclaims=%d vertices_reclaimed=%ld peels=%d\n",algo.c_str(),nReclaim,nReclaimed,nPeel);
  fprintf(stderr,"[b3] gamma_tested=%ld  O_t_nonempty=%ld  hits(w_bar=0)=%ld  O_t_max=%ld\n",
          statGammaTested,statGammaNonEmptyOt,statGammaZeroed,statOtMax);
  fprintf(stderr,"[dylo:%s] pruned=%ld deferred=%ld flushes=%ld (vanished=%ld quality=%ld)\n",algo.c_str(),statPruned,statDeferred,statFlushed,statFlushVanished,statFlushQuality);
  fprintf(stderr,"[dylo:%s] mean_vol(A)/|E| = %.5f over %ld rounds\n",algo.c_str(),
          statRegionBatches?statVolA/statRegionBatches:0.0,statRegionBatches);
  fprintf(stderr,"[dylo:%s] recheck_run=%ld recheck_skipped=%ld (%.1f%% skipped)\n",
          algo.c_str(),statRecheckRun,statRecheckSkipped,
          (statRecheckRun+statRecheckSkipped)?100.0*statRecheckSkipped/(statRecheckRun+statRecheckSkipped):0.0);
  return 0;
}
