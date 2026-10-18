// DyLo base modularity-optimization detector (Algorithm 1) + plug-in API.
// Self-contained C++17. Weighted undirected graph with incremental edge insertion.
// Instances differ ONLY in Refine: Louvain=identity, SLM=local-move-within-community,
// Leiden=well-connected split. Detect(G) builds the community hierarchy T (levels).
#pragma once
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>

namespace dylo {

using K = uint32_t;
enum class Instance { LOUVAIN, SLM, LEIDEN };

// Weighted undirected graph (adjacency with self-loop weight). Supports incremental
// edge insertion (for the dynamic driver) and aggregation.
struct Graph {
  int n = 0;
  std::vector<std::unordered_map<K,double>> adj;  // adj[u][v] = w (u!=v), symmetric
  std::vector<double> self;                        // self-loop weight (from aggregation)
  std::vector<double> k;                           // weighted degree (self counts twice)
  double m2 = 0.0;                                 // sum_i k[i] = 2m
  // Flat append-only adjacency (CSR-like) for cache-friendly iteration in the hot
  // affected-region histogram. Mirrors adj for SIMPLE insertion-only streams (no duplicate
  // edges -> exact). modularity()/local_moving() still use adj, so quality is unaffected even
  // if a stream did contain duplicates. Built lazily (only when fadj_on).
  bool fadj_on = false;
  std::vector<std::vector<std::pair<int,float>>> fadj;

  void ensure(int N){ if(N>n){ adj.resize(N); self.resize(N,0.0); k.resize(N,0.0); if(fadj_on) fadj.resize(N); n=N; } }
  void enable_fadj(){ fadj_on=true; fadj.resize(n);
    for(int u=0;u<n;u++){ fadj[u].clear(); for(auto&[v,w]:adj[u]) fadj[u].emplace_back(v,(float)w); } }
  // Rebuild the unordered_map adjacency from fadj (for the untimed audit/rebuild, which use
  // adj-based local_moving/detect). Called lazily, off the hot path.
  void rebuild_adj(){ for(int u=0;u<n;u++){ adj[u].clear(); for(auto&[v,w]:fadj[u]) adj[u][(K)v]+=w; } }
  void add_edge(K u, K v, double w){
    ensure(std::max(u,v)+1);
    if(u==v){ self[u]+=w; k[u]+=2*w; m2+=2*w; return; }
    k[u]+=w; k[v]+=w; m2+=2*w;
    if(fadj_on){ fadj[u].emplace_back((int)v,(float)w); fadj[v].emplace_back((int)u,(float)w); } // hot path: fadj only
    else { adj[u][v]+=w; adj[v][u]+=w; }                  // adj maintained only when fadj is off
  }
};

// Modularity over the flat fadj adjacency (used by the fast incremental path; identical
// value to modularity() for simple graphs, avoids touching the unordered_map adjacency).
// Objective switch. Modularity's null model couples every community through the
// global edge weight m: inserting ANY edge changes every community's term, which is
// what forces the M2 drift sweep in the certificate scheme. CPM
//     H = sum_c [ e_c - gamma * n_c(n_c-1)/2 ]
// has no such term -- an internal insertion touches one community, a crossing
// insertion touches none -- and its move gain is k_vc - gamma*n_c, which depends on
// the destination alone. gamma is NOT comparable to modularity's resolution: it is an
// absolute density threshold, so it has to be calibrated per graph.
inline bool use_cpm(){
  static const bool B = [](){ const char* e=std::getenv("DYLO_OBJ"); return e && std::string(e)=="cpm"; }();
  return B;
}
inline double cpm_gamma(){
  static const double G = [](){ const char* e=std::getenv("DYLO_CPM_GAMMA"); return e?std::atof(e):1e-4; }();
  return G;
}

// CPM objective value of a partition.
inline double cpm_value(const Graph& g, const std::vector<int>& comm, double gamma){
  std::unordered_map<int,double> ein; std::unordered_map<int,long> nc;
  for(int v=0;v<g.n;v++){
    if(g.k[v]<=0) continue;
    nc[comm[v]]++;
    for(auto&[u,w]:g.adj[v]) if(comm[(int)u]==comm[v] && (int)u!=v) ein[comm[v]]+=w;
  }
  double H=0;
  for(auto&[c,n]:nc){ double e=(ein.count(c)?ein[c]:0.0)/2.0; H += e - gamma*(double)n*(n-1)/2.0; }
  return H;
}

inline double modularity_fadj(const Graph& g, const std::vector<int>& comm){
  if(g.m2==0) return 0.0;
  std::unordered_map<int,double> in, tot;
  for(int u=0;u<g.n;u++){
    tot[comm[u]] += g.k[u];
    in[comm[u]]  += 2.0*g.self[u];
    for(auto&[v,w]:g.fadj[u]) if(comm[u]==comm[v]) in[comm[u]] += w;   // each intra edge twice
  }
  double q=0.0; for(auto&[c,t]:tot){ double i=in.count(c)?in[c]:0.0; q += i/g.m2 - (t/g.m2)*(t/g.m2); }
  return q;
}

// Modularity of a membership over g.
inline double modularity(const Graph& g, const std::vector<int>& comm){
  if(g.m2==0) return 0.0;
  // A partition vector is sized when its hierarchy is built; the graph can gain
  // vertices afterwards, which it does on any stream where entities arrive over
  // time. Such a vertex is not yet in any community, so it is read as its own --
  // the alternative, indexing past the end, is a heap over-read that corrupts
  // the allocator and surfaces much later as a malloc assertion.
  const int NC=(int)comm.size();
  auto cid=[&](int x)->int{ return x<NC ? comm[x] : -1-x; };
  std::unordered_map<int,double> in, tot;
  for(int u=0;u<g.n;u++){
    const int cu=cid(u);
    tot[cu] += g.k[u];
    in[cu] += 2.0*g.self[u];
    for(auto&[v,w]:g.adj[u]) if(cu==cid((int)v)) in[cu] += w; // each intra edge twice
  }
  double q=0.0; for(auto&[c,t]:tot){ double i=in.count(c)?in[c]:0.0; q += i/g.m2 - (t/g.m2)*(t/g.m2); }
  return q;
}

// One full local-moving optimisation on g starting from comm (mutated).
// Returns true if any vertex moved. gamma = resolution (1.0).
inline bool local_moving(const Graph& g, std::vector<int>& comm, std::mt19937& rng, double gamma=1.0){
  if(g.n<=1||g.m2==0) return false;
  std::unordered_map<int,double> ctot;
  for(int u=0;u<g.n;u++) ctot[comm[u]] += g.k[u];
  const bool CPM = use_cpm(); const double CG = cpm_gamma();
  std::unordered_map<int,long> csz;
  if(CPM) for(int u=0;u<g.n;u++) csz[comm[u]]++;
  std::vector<int> order(g.n); for(int i=0;i<g.n;i++) order[i]=i;
  bool moved=false, improved=true;
  while(improved){
    improved=false; std::shuffle(order.begin(),order.end(),rng);
    for(int u:order){
      int ci=comm[u]; double ku=g.k[u];
      std::unordered_map<int,double> kin;
      for(auto&[v,w]:g.adj[u]) kin[comm[v]]+=w;
      ctot[ci]-=ku; if(CPM) csz[ci]--;
      auto gainOf=[&](int c,double kic)->double{
        return CPM ? (kic - CG*(double)csz[c]) : (kic - gamma*ctot[c]*ku/g.m2);
      };
      int best=ci; double bestgain=gainOf(ci, kin.count(ci)?kin[ci]:0.0);
      for(auto&[c,kic]:kin){ if(c==ci) continue;
        double gn=gainOf(c,kic); if(gn>bestgain+1e-12){bestgain=gn;best=c;} }
      ctot[best]+=ku; if(CPM) csz[best]++;
      if(best!=ci){ comm[u]=best; moved=true; improved=true; }
    }
  }
  return moved;
}

inline int renumber(std::vector<int>& comm){
  std::unordered_map<int,int> r; for(int& c:comm){ auto it=r.find(c); if(it==r.end()){int id=r.size(); r[c]=id; c=id;} else c=it->second; }
  return (int)r.size();
}

// Refine within each community (instance-specific). Returns sub-membership (0..S-1):
//  LOUVAIN: identity (each community is its own sub-community);
//  SLM (Waltman & van Eck 2013): full local-moving WITHIN each community from singletons,
//    iterated to convergence -> sub-communities; combined with detect()'s aggregate-init
//    lift this is Smart Local Moving (sub-communities of a community start together);
//  LEIDEN (Traag, van Eck & Waltman 2019): single-pass, connectedness-gated merge from
//    singletons -> guarantees well-connected sub-communities.
inline std::vector<int> refine(const Graph& g, const std::vector<int>& comm, int ncomm,
                               Instance inst, std::mt19937& rng, double gamma=1.0){
  std::vector<int> sub(g.n);
  if(inst==Instance::LOUVAIN){ for(int i=0;i<g.n;i++) sub[i]=comm[i]; return sub; }
  if(inst==Instance::SLM){
    // SLM refine: within each community, run full local-moving from singletons to
    // convergence (moves are confined to the community's own subgraph). Over-splitting is
    // recovered by detect()'s lift, which lets sub-communities of one community re-merge.
    std::vector<std::vector<int>> members(ncomm);
    for(int i=0;i<g.n;i++) members[comm[i]].push_back(i);
    int next=0;
    for(int c=0;c<ncomm;c++){
      auto& nodes=members[c];
      if(nodes.size()<=1){ for(int v:nodes) sub[v]=next++; continue; }
      std::unordered_map<int,int> sc;                    // node -> local sub-comm id
      std::unordered_map<int,double> sigma;              // sub-comm -> total degree
      int sid=0; for(int v:nodes){ sc[v]=sid; sigma[sid]=g.k[v]; sid++; }
      std::vector<int> order=nodes; bool improved=true;
      while(improved){
        improved=false; std::shuffle(order.begin(),order.end(),rng);
        for(int v:order){
          double kv=g.k[v];
          std::unordered_map<int,double> kin;
          for(auto&[w,wt]:g.adj[v]) if(comm[w]==c) kin[sc[w]]+=wt;
          int cur=sc[v]; sigma[cur]-=kv;
          double cgain=(kin.count(cur)?kin[cur]:0.0) - gamma*sigma[cur]*kv/g.m2;
          int best=cur; double bestgain=cgain;
          for(auto&[t,kit]:kin){ if(t==cur) continue;
            double gn=kit-gamma*sigma[t]*kv/g.m2; if(gn>bestgain+1e-12){bestgain=gn;best=t;} }
          sigma[best]+=kv;
          if(best!=cur){ sc[v]=best; improved=true; }
        }
      }
      std::unordered_map<int,int> remap; for(int v:nodes){ int s=sc[v]; if(!remap.count(s)) remap[s]=next++; sub[v]=remap[s]; }
    }
    return sub;
  }
  // Faithful Leiden/SLM refinement (Traag, van Eck & Waltman 2019): within each
  // community, start from singletons and merge each WELL-CONNECTED node into the
  // best positive-gain sub-community in a SINGLE pass (only singletons move). Unlike
  // full within-community local-moving (which iterates to convergence and over-splits),
  // this keeps a well-connected community as one sub-community while splitting off
  // weakly-connected nodes -> preserves Leiden's connectedness without fragmenting.
  std::vector<std::vector<int>> members(ncomm);
  for(int i=0;i<g.n;i++) members[comm[i]].push_back(i);
  int next=0;
  for(int c=0;c<ncomm;c++){
    auto& nodes=members[c];
    if(nodes.size()<=1){ for(int v:nodes) sub[v]=next++; continue; }
    double kC=0; for(int v:nodes) kC+=g.k[v];
    std::unordered_map<int,int> sc;                    // node -> local sub-comm id
    std::unordered_map<int,double> sigma;              // sub-comm -> total degree
    int sid=0; for(int v:nodes){ sc[v]=sid; sigma[sid]=g.k[v]; sid++; }
    std::unordered_map<int,int> ssize; for(auto&[v,s]:sc) ssize[s]++;
    std::vector<int> order=nodes; std::shuffle(order.begin(),order.end(),rng);
    for(int v:order){
      if(ssize[sc[v]]!=1) continue;                    // only move singletons
      double kv=g.k[v];
      // edges from v to each sub-comm within c, and total edges to rest of c (E(v,C\v))
      std::unordered_map<int,double> kin; double EvC=0;
      for(auto&[w,wt]:g.adj[v]) if(comm[w]==c){ kin[sc[w]]+=wt; EvC+=wt; }
      // well-connectedness gate: v sufficiently tied to its community
      if(EvC < gamma*kv*(kC-kv)/g.m2) continue;         // weakly-connected -> stays singleton
      int my=sc[v]; sigma[my]-=kv; ssize[my]--;
      int best=-1; double bestgain=0.0;                 // require strictly positive gain
      for(auto&[t,kit]:kin){ if(t==my) continue;
        double gn=kit-gamma*sigma[t]*kv/g.m2; if(gn>bestgain+1e-12){bestgain=gn;best=t;} }
      if(best>=0){ sc[v]=best; sigma[best]+=kv; ssize[best]++; }
      else { sigma[my]+=kv; ssize[my]++; }              // no improving merge -> singleton
    }
    std::unordered_map<int,int> remap; for(int v:nodes){ int s=sc[v]; if(!remap.count(s)) remap[s]=next++; sub[v]=remap[s]; }
  }
  return sub;
}

// Aggregate g by sub-membership (0..S-1) -> reduced graph.
inline Graph aggregate(const Graph& g, const std::vector<int>& sub, int S){
  Graph r; r.ensure(S);
  for(int u=0;u<g.n;u++){
    int su=sub[u]; r.self[su]+=g.self[u];
    for(auto&[v,w]:g.adj[u]){ int sv=sub[v]; if(su==sv) r.self[su]+=w/2.0; else r.adj[su][sv]+=w; }
  }
  r.k.assign(S,0.0); r.m2=0.0;
  for(int s=0;s<S;s++){ double d=2*r.self[s]; for(auto&[v,w]:r.adj[s]) d+=w; r.k[s]=d; r.m2+=d; }
  return r;
}

// Hierarchy: levels[i] = membership over ORIGINAL vertices at aggregation pass i (L0 finest).
struct Hierarchy { std::vector<std::vector<int>> levels; double Q=0.0; };

// Detect(G) — Algorithm 1: repeat (local-move; Refine; Aggregate) until Q stops rising.
inline Hierarchy detect(const Graph& g0, Instance inst, uint32_t seed=42, double gamma=1.0,
                        const std::vector<int>* init0=nullptr){
  std::mt19937 rng(seed);
  Hierarchy H;
  Graph g = g0;                              // working graph (shrinks each pass)
  std::vector<int> base(g0.n); for(int i=0;i<g0.n;i++) base[i]=i; // orig -> current-node
  std::vector<int> nextInit;                 // Leiden lift: init partition for next level (empty => singletons)
  bool first=true;
  while(true){
    std::vector<int> comm(g.n);
    if(first && init0){ comm=*init0; renumber(comm); }              // iterate from prior partition (outer)
    else if(!nextInit.empty()){ comm=nextInit; renumber(comm); }    // Leiden: start from lifted communities P
    else for(int i=0;i<g.n;i++) comm[i]=i;                          // singletons (level 0)
    first=false;
    local_moving(g, comm, rng, gamma);                              // MoveNodesFast
    int nc = renumber(comm);
    // record this level's flat partition over ORIGINAL vertices
    std::vector<int> lvl(g0.n); for(int i=0;i<g0.n;i++) lvl[i]=comm[base[i]];
    H.levels.push_back(lvl);
    if(nc==g.n) break;                        // every node its own community -> converged
    // Refine communities into sub-communities (instance-specific) for aggregation
    std::vector<int> sub = refine(g, comm, nc, inst, rng, gamma);
    int S = renumber(sub);
    if(S>=g.n) break;                         // refinement didn't reduce -> done
    // Leiden lift: each aggregate node (a refined sub-community) keeps its community P
    std::vector<int> lift(S); for(int u=0;u<g.n;u++) lift[sub[u]]=comm[u];
    g = aggregate(g, sub, S);
    for(int i=0;i<g0.n;i++) base[i]=sub[base[i]];
    nextInit.swap(lift);
  }
  if(H.levels.empty()){ std::vector<int> lvl(g0.n); for(int i=0;i<g0.n;i++) lvl[i]=i; H.levels.push_back(lvl); }
  H.Q = modularity(g0, H.levels.back());
  return H;
}

// Build a multi-level dendrogram with L0 FIXED to a given leaf partition (the
// incrementally-maintained finest communities). Unlike detect(), level 0 is NOT
// re-optimised from singletons — it is taken as given, and only the COARSER levels
// (L1..top) are (re)built by aggregation + local-moving. Used by the cost-aware driver
// to maintain a genuine summary hierarchy on top of the incremental leaf partition
// cheaply (the aggregated graphs are small). levels[0]=leaf, levels.back()=top.
inline Hierarchy build_levels_from_leaf(const Graph& g0, std::vector<int> leaf,
                                        Instance inst, std::mt19937& rng, double gamma=1.0){
  Hierarchy H;
  int n0 = renumber(leaf);
  H.levels.push_back(leaf);                                  // L0 = given finest partition
  if(n0<=1){ H.Q = modularity(g0, H.levels.back()); return H; }
  Graph g = aggregate(g0, leaf, n0);                          // nodes = leaf communities
  std::vector<int> base(g0.n); for(int i=0;i<g0.n;i++) base[i]=leaf[i]; // orig -> current node
  std::vector<int> nextInit;
  while(true){
    std::vector<int> comm(g.n);
    if(!nextInit.empty()){ comm=nextInit; renumber(comm); }
    else for(int i=0;i<g.n;i++) comm[i]=i;                    // singleton aggregate nodes
    local_moving(g, comm, rng, gamma);
    int nc = renumber(comm);
    if(nc==g.n) break;                                        // nothing merged -> top reached
    std::vector<int> lvl(g0.n); for(int i=0;i<g0.n;i++) lvl[i]=comm[base[i]];
    H.levels.push_back(lvl);
    std::vector<int> sub = refine(g, comm, nc, inst, rng, gamma);
    int S = renumber(sub);
    if(S>=g.n) break;
    std::vector<int> lift(S); for(int u=0;u<g.n;u++) lift[sub[u]]=comm[u];
    g = aggregate(g, sub, S);
    for(int i=0;i<g0.n;i++) base[i]=sub[base[i]];
    nextInit.swap(lift);
  }
  H.Q = modularity(g0, H.levels.back());
  return H;
}

// Iterated Detect: re-run from the previous flat partition until Q stops improving.
// leidenalg / SLM run multiple iterations; a single pass leaves a small residual gap.
// maxIter caps work; in practice this converges in 2-3 iterations.
inline Hierarchy detect_iter(const Graph& g0, Instance inst, uint32_t seed=42,
                             double gamma=1.0, int maxIter=8){
  Hierarchy H = detect(g0, inst, seed, gamma);
  for(int it=1; it<maxIter; it++){
    std::vector<int> prev = H.levels.back();
    Hierarchy H2 = detect(g0, inst, seed+it, gamma, &prev);
    if(H2.Q <= H.Q + 1e-9) break;            // no further improvement -> stable
    H = H2;
  }
  return H;
}

// Top-down (GraphRAG-style) hierarchy. Bottom-up aggregation AT THE SAME
// RESOLUTION cannot build one here: the given partition is already a modularity
// optimum, so merging any two of its communities lowers Q and
// build_levels_from_leaf returns ONE level.
//
// An earlier version of this note claimed aggregation was impossible at ANY
// resolution, on the grounds that two communities sharing no edge have merge
// gain zero even as gamma->0. That argument is sound but covers only the
// ISOLATED communities; the ones that do share an edge merge readily once
// gamma drops. Measured: Facebook goes from 1 level at gamma=1 to 3 at
// gamma=0.1 (18/8/8), Epinions from 1 to 4 (1037/308/308/308). The note also
// quoted "4008 of 4742 communities isolated" on Epinions -- those counts
// included zero-degree vertices as communities of their own; over vertices
// that actually have edges it is 1037 communities. So resolution-aggregation
// IS a viable way to build the tree (DYLO_HIER_GAMMA in driver_td5), and it
// has one property splitting lacks: the maintained partition ends up at
// levels[0], i.e. the summary-tree LEAVES, so an action's anchor weight and
// the destination community's size are the same number.
//
// What splitting still buys is leaf granularity. Under aggregation the leaves
// are the maintained communities themselves (~224 members on Facebook), so
// dirtying one costs its whole membership; splitting drives leaves down to
// maxSize (~10), which is an order of magnitude cheaper per dirty leaf. That
// trade -- unified anchor definition vs. cheaper leaves -- is what the two
// modes are for.
//
// Splitting works inside a single community and needs no inter-community
// edges. `coarse` (the maintained partition) becomes the TOP level and any
// community above maxSize is recursively re-clustered. Returned finest-first,
// so levels[0] are the summary-tree leaves and levels.back() is the maintained
// partition, matching Hierarchy's convention.
inline Hierarchy build_levels_topdown(const Graph& g0, std::vector<int> coarse,
                                      Instance inst, std::mt19937& rng,
                                      int maxSize=25, int maxDepth=6){
  std::vector<std::vector<int>> desc;
  renumber(coarse);
  desc.push_back(coarse);
  std::vector<int> cur = coarse;
  for(int d=0; d<maxDepth; d++){
    std::unordered_map<int,std::vector<int>> bag;
    for(int v=0; v<g0.n; v++) if(g0.k[v]>0) bag[cur[v]].push_back(v);
    std::vector<int> next = cur;
    int nextId = 0; for(int v=0;v<g0.n;v++) nextId=std::max(nextId,cur[v]+1);
    bool anySplit=false;
    for(auto&[c,vs] : bag){
      if((int)vs.size() <= maxSize) continue;
      std::unordered_map<int,int> loc; loc.reserve(vs.size());
      for(size_t i=0;i<vs.size();i++) loc[vs[i]]=(int)i;
      Graph sub;
      for(int v : vs) for(auto&[u,w] : g0.adj[v]){
        auto it=loc.find((int)u);
        if(it!=loc.end() && loc[v] <= it->second)
          sub.add_edge((uint32_t)loc[v],(uint32_t)it->second,w);
      }
      if(sub.n<2) continue;
      Hierarchy hs = detect_iter(sub, inst, 42);
      std::vector<int> sp = hs.levels[0];
      int ns = renumber(sp);
      if(ns<2) continue;                       // indivisible
      anySplit=true;
      // `sub` is built from the community's INTERNAL edges only, so a member
      // with no neighbour inside the community never appears in it and
      // sub.n < vs.size(). Indexing sp by loc[v] then reads past the end --
      // an out-of-bounds read confirmed by ASan on cit-hepth/dynamo. Such a
      // member is genuinely unsplittable within this community, so it becomes
      // its own fine leaf.
      std::vector<int> newid(ns);
      for(int i=0;i<ns;i++) newid[i]=nextId++;
      for(int v : vs){
        int li2 = loc[v];
        next[v] = (li2 < (int)sp.size()) ? newid[sp[li2]] : nextId++;
      }
    }
    if(!anySplit) break;
    renumber(next);
    desc.push_back(next);
    cur = next;
  }
  Hierarchy H;
  H.levels.assign(desc.rbegin(), desc.rend());
  H.Q = modularity(g0, H.levels.back());
  return H;
}

// Both directions from the same starting partition: split it DOWN into fine
// leaves and aggregate it UP at a lower resolution, then stack the two into one
// tree. levels[0] are the split leaves, the starting partition sits at an
// interior level, levels.back() is the coarsest aggregate.
//
// Each direction alone gives one of the two properties and not the other.
// Splitting yields cheap leaves (~maxSize members) but leaves the MAINTAINED
// partition at levels.back(), so the size prefilter measures a community while
// the action anchors a leaf -- the filter is stricter than the model. Pure
// aggregation puts the maintained partition at levels[0], where prefilter and
// anchor agree, but then whole communities ARE the leaves: measured 6.5-14.6x
// more expensive to re-summarize (Facebook 0.0680 vs 0.0064, Cit-HepTh 0.0507
// vs 0.0035). Stacking both puts the maintained partition at levels[0] AND
// keeps the leaves small.
//
// The price is that the maintained partition becomes the FINE one, which
// changes what the index exposes: modularity is then read at the leaf level,
// where splitting has by construction made it lower than at the community
// level. That trade is the thing to measure, not to assume.
//
// The two halves share the given partition, so `up`'s level 0 is dropped and
// `down`'s last level plays that role. Labels need not agree between them:
// make_cost_tree derives each child's parent from a member vertex's label at
// the level above, so any relabelling of the same partition is equivalent.
inline Hierarchy build_levels_combined(const Graph& g0, std::vector<int> coarse,
                                       Instance inst, std::mt19937& rng,
                                       int maxSize=25, double gamma=0.5, int maxDepth=6){
  Hierarchy down = build_levels_topdown(g0, coarse, inst, rng, maxSize, maxDepth);
  Hierarchy up   = build_levels_from_leaf(g0, coarse, inst, rng, gamma);
  Hierarchy H;
  H.levels = down.levels;                                   // finest .. the given partition
  for(size_t i=1;i<up.levels.size();i++) H.levels.push_back(up.levels[i]);
  H.Q = modularity(g0, H.levels[0]);                        // maintained level is now the FINEST
  return H;
}

} // namespace dylo
