#include "e1_gate.hpp"
// HIT-Leiden (Hierarchical Incremental Tree Leiden), reimplemented from
// arXiv:2601.08554 (Fang et al., ByteDance) against this harness's baseline
// CLI so it can be run side by side with DyLo and the other baselines:
//
//   ./hit_leiden_driver <initial.txt> <inc-base> <numBatches> <outDir>
//
// The paper's algorithm, as implemented here:
//   inc-movement   (Alg. 2) seeds the affected set A with the endpoints of
//                  cross-community insertions and intra-community deletions,
//                  then repeatedly pops v, moves it to argmax_C DeltaQ(v->C)
//                  when that gain is positive, and pushes the neighbours that
//                  are outside the target community. Sub-community edges are
//                  tracked in G_Psi so that a deletion which disconnects a
//                  sub-community is recorded in K.
//   inc-refinement (Alg. 3) re-labels each split connected component of G_Psi
//                  as a fresh sub-community, then tries to absorb singleton
//                  sub-communities into the best locally-optimal neighbouring
//                  sub-community within the same community.
//   inc-aggregation(Sec. V-C) turns edge changes and sub-community migrations
//                  into superedge changes for the next level.
//   def-update     (Alg. 5) walks levels P..1 propagating community labels
//                  down through the supervertex mappings.
//
// Deviations, all forced by this harness rather than by the algorithm, are
// noted where they occur; the stream here is insertion-only (see
// prep_data.py), so the deletion branches are implemented but never exercised
// by these datasets.
#include "detector.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <queue>
#include <unordered_map>
#include <unordered_set>
using namespace dylo;

static std::string extend(const std::string& p, const std::string& e){
  size_t i=p.rfind('.'); return p.substr(0,i)+e+"."+p.substr(i+1); }

template<class F> static void read_edges(const std::string& path, F&& cb){
  std::ifstream f(path, std::ios::binary);
  if(!f) return;
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const char* p=buf.data(); const char* end=p+buf.size();
  while(p<end){
    while(p<end && (*p=='\n'||*p=='\r'||*p==' '||*p=='\t')) ++p;
    if(p>=end) break;
    if(*p=='#'){ while(p<end && *p!='\n') ++p; continue; }
    long u=0; bool any=false; while(p<end && *p>='0'&&*p<='9'){ u=u*10+(*p-'0'); ++p; any=true; }
    if(!any){ while(p<end && *p!='\n') ++p; continue; }
    while(p<end && (*p==' '||*p=='\t')) ++p;
    long v=0; while(p<end && *p>='0'&&*p<='9'){ v=v*10+(*p-'0'); ++p; }
    while(p<end && *p!='\n') ++p;
    cb((K)u,(K)v,1.0);
  }
}

int main(int argc,char**argv){
  if(argc<5){
    fprintf(stderr,"usage: %s <initial.txt> <inc-base> <numBatches> <outDir> [gamma]\n"
                   "  HIT-Leiden (arXiv:2601.08554). P levels via DYLO_P (default 3).\n",argv[0]);
    return 2;
  }
  std::string initial=argv[1], incBase=argv[2];
  int nB=atoi(argv[3]);
  std::string outDir=argv[4];
  const double gamma = (argc>5)? atof(argv[5]) : 1.0;
  const char* pEnv=std::getenv("DYLO_P");
  const int P = pEnv? std::max(1,atoi(pEnv)) : 3;
  std::filesystem::create_directories(outDir);

  Graph g; read_edges(initial,[&](uint32_t u,uint32_t v,double w){ g.add_edge(u,v,w); });
  fprintf(stderr,"[hit-leiden] initial %d V, m=%.0f, P=%d, gamma=%.2f\n",g.n,g.m2/2,P,gamma);
  std::mt19937 rng(e1seed());

  // Level-0 state: community f(.) and sub-community s(.), initialised by one
  // full Leiden run (the paper likewise assumes an existing partition to
  // maintain -- it maintains, it does not bootstrap).
  Hierarchy H0 = detect_iter(g, Instance::LEIDEN, e1seed(), gamma);
  std::vector<int> f = H0.levels[0];
  renumber(f);
  std::vector<int> s = f;                 // sub-community starts equal to community
  int nextSub = 0; for(int v=0;v<g.n;v++) nextSub=std::max(nextSub,s[v]+1);

  // ctot[c] = total incident weight of community c (for the DeltaQ formula)
  std::unordered_map<int,double> ctot;
  for(int v=0;v<g.n;v++) ctot[f[v]] += g.k[v];

  // Persistent level-1 supergraph over sub-communities, plus the previous
  // step's vertex->supervertex mapping s_pre. Both are carried across
  // batches; inc-aggregation mutates Hsup by superedge deltas only.
  std::vector<int> sPre = s;
  Graph Hsup;
  for(int v=0; v<g.n; v++)
    for(auto&[u,w] : g.adj[v])
      if((int)u >= v) Hsup.add_edge((K)sPre[v], (K)sPre[(int)u], w);

  auto write_com=[&](int k,const std::vector<int>& memb){
    std::unordered_map<int,std::vector<int>> bag;
    for(int i=0;i<(int)memb.size();i++) if(g.k[i]>0) bag[memb[i]].push_back(i);
    std::ofstream o(outDir+"/com_"+std::to_string(k)+".txt");
    for(auto&[c,vs]:bag){ for(size_t i=0;i<vs.size();i++){ if(i)o<<'\t'; o<<vs[i]; } o<<'\n'; }
  };
  auto count_comm=[&](const std::vector<int>& memb){
    std::unordered_set<int> t; for(int i=0;i<(int)memb.size();i++) if(g.k[i]>0) t.insert(memb[i]);
    return (int)t.size(); };

  std::ofstream metrics(outDir+"/metrics.tsv"); metrics<<std::fixed<<std::setprecision(6);
  metrics<<"batch\tlatency_ms\tmodularity\tnum_communities\trounds\n";
  metrics<<"0\t0.000000\t"<<modularity(g,f)<<"\t"<<count_comm(f)<<"\t0\n"; metrics.flush();
  write_com(0,f);

  e1gate();
  for(int b=1;b<=nB;b++){
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<std::pair<uint32_t,uint32_t>> ins;
    read_edges(extend(incBase,"_"+std::to_string(b)),[&](uint32_t u,uint32_t v,double w){
      g.add_edge(u,v,w); ins.emplace_back(u,v); });

    // grow state for any new vertices: each starts as its own community and
    // its own sub-community (level-0 singletons)
    if((int)f.size()<g.n){
      int nc=0; for(int c:f) nc=std::max(nc,c+1);
      for(int v=(int)f.size(); v<g.n; v++){ f.push_back(nc++); s.push_back(nextSub++); }
    }
    for(int v=0;v<g.n;v++) if(!ctot.count(f[v])) ctot[f[v]]=0.0;
    // ctot must reflect the batch's degree growth before any DeltaQ is taken
    for(auto&[u,v]:ins){
      if(u==v){ ctot[f[u]] += 2.0; continue; }
      ctot[f[u]] += 1.0; ctot[f[v]] += 1.0;
    }

    // ---- inc-movement (Alg. 2) ----
    std::vector<char> inA(g.n,0);
    std::vector<int> A;
    std::unordered_set<int> B, Kset;
    auto pushA=[&](int v){ if(v<g.n && !inA[v]){ inA[v]=1; A.push_back(v); } };
    for(auto&[u,v]:ins){
      if(u==v) continue;
      if(f[u]!=f[v]){ pushA((int)u); pushA((int)v); }   // cross-community insertion
      // insertion never disconnects G_Psi, so K is untouched here; the
      // deletion branch (alpha<0) of Alg. 2 has no counterpart in this
      // insertion-only harness.
    }
    size_t head=0;
    while(head<A.size()){
      int v=A[head++]; inA[v]=0;
      double kv=g.k[v];
      int cv=f[v];
      std::unordered_map<int,double> kin;
      for(auto&[u,w]:g.adj[v]) if((int)u!=v) kin[f[u]]+=w;
      ctot[cv]-=kv;
      double stay=(kin.count(cv)?kin[cv]:0.0) - gamma*ctot[cv]*kv/g.m2;
      int best=cv; double bestg=stay;
      for(auto&[c,kic]:kin){
        if(c==cv) continue;
        double gn=kic - gamma*ctot[c]*kv/g.m2;
        if(gn>bestg+1e-12){ bestg=gn; best=c; }
      }
      // "C union {empty}" in line 11: moving out to a fresh singleton
      if(0.0 > bestg+1e-12){ bestg=0.0; best=-1; }
      if(best==-1){ int nc=0; for(auto&[c,t]:ctot) nc=std::max(nc,c+1); best=nc; ctot[best]=0.0; }
      ctot[best]+=kv;
      if(best!=cv){
        f[v]=best; B.insert(v); Kset.insert(v);
        for(auto&[u,w]:g.adj[v]) if((int)u!=v && f[u]!=best) pushA((int)u);
      }
    }

    // ---- inc-refinement (Alg. 3) ----
    // Re-label each connected component of G_Psi (intra-sub-community edges)
    // touched via K, then absorb singleton sub-communities.
    std::vector<int> R;
    {
      std::unordered_set<int> seen;
      for(int v0 : Kset){
        if(seen.count(v0)) continue;
        // BFS over intra-sub-community, intra-community edges
        std::vector<int> comp; std::queue<int> q; q.push(v0); seen.insert(v0);
        while(!q.empty()){
          int x=q.front(); q.pop(); comp.push_back(x);
          for(auto&[y,w]:g.adj[x]){
            int yy=(int)y;
            if(yy>=g.n || seen.count(yy)) continue;
            if(s[yy]==s[x] && f[yy]==f[x]){ seen.insert(yy); q.push(yy); }
          }
        }
        int ns=nextSub++;
        for(int x:comp){ s[x]=ns; R.push_back(x); }
      }
      // sizes of sub-communities, to find singletons
      std::unordered_map<int,int> ssz;
      for(int v=0;v<g.n;v++) if(g.k[v]>0) ssz[s[v]]++;
      std::unordered_map<int,double> stot;
      for(int v=0;v<g.n;v++) stot[s[v]] += g.k[v];
      for(int v : R){
        if(ssz[s[v]]!=1) continue;
        double kv=g.k[v];
        std::unordered_map<int,double> sin;
        for(auto&[u,w]:g.adj[v]) if((int)u!=v && f[u]==f[v]) sin[s[u]]+=w;
        int bestS=-1; double bestg=1e-12;
        for(auto&[sc,kic]:sin){
          if(sc==s[v]) continue;
          double gn=kic - gamma*stot[sc]*kv/g.m2;
          if(gn>bestg){ bestg=gn; bestS=sc; }
        }
        if(bestS>=0){
          stot[s[v]]-=kv; ssz[s[v]]--;
          s[v]=bestS; stot[bestS]+=kv; ssz[bestS]++;
        }
      }
    }

    // ---- inc-aggregation (Alg. 4) ----
    // The supergraph H over sub-communities is PERSISTENT across batches and
    // updated by superedge deltas only, exactly as Alg. 4 prescribes:
    //   (a) every raw edge change (vi,vj,alpha) contributes (s_pre(vi),
    //       s_pre(vj), alpha), because a superedge weight is the sum of the
    //       edge weights between the two sub-communities;
    //   (b) every vertex vi that changed sub-community (vi in R) removes its
    //       incident weight under s_pre and re-adds it under s_cur, with the
    //       "s_cur(vj)==s_pre(vj) or i<j" guard of line 7 so a pair of movers
    //       is not double-counted;
    //   (c) self-loops move with the vertex (lines 10-11).
    // Rebuilding H from scratch each batch (as the first version of this
    // driver did) yields the same partition, but its cost is O(|E|) per
    // batch, which is precisely the cost the paper's design exists to avoid
    // -- so a latency comparison against it would be measuring the
    // reimplementation, not the algorithm.
    {
      std::vector<std::tuple<int,int,double>> dH;
      for(auto&[u,v]:ins){
        int su=sPre[u], sv=sPre[v];
        dH.emplace_back(su, sv, 1.0);
      }
      std::vector<char> inR(g.n,0);
      for(int v: R) if(v<g.n) inR[v]=1;
      for(int vi=0; vi<g.n; vi++){
        if(!inR[vi]) continue;
        for(auto&[vj_,w]:g.adj[vi]){
          int vj=(int)vj_;
          if(vj==vi) continue;
          const bool vjStayed = (vj<(int)sPre.size()) && (s[vj]==sPre[vj]);
          if(vjStayed || vi<vj){
            dH.emplace_back(sPre[vi], sPre[vj], -w);
            dH.emplace_back(s[vi],    s[vj],     w);
          }
        }
        if(g.self[vi]>0){
          dH.emplace_back(sPre[vi], sPre[vi], -g.self[vi]);
          dH.emplace_back(s[vi],    s[vi],     g.self[vi]);
        }
      }
      // Compress(dH) and apply to the persistent H.
      std::unordered_map<long long,double> comp;
      auto key=[&](int a,int b){ if(a>b) std::swap(a,b); return ((long long)a<<32)|(unsigned)b; };
      for(auto&[a,b,w]:dH) comp[key(a,b)] += w;
      for(auto&[k2,w]:comp){
        if(std::fabs(w)<1e-12) continue;
        int a=(int)(k2>>32), b=(int)(k2&0xffffffff);
        Hsup.add_edge((K)a,(K)b,w);
      }
      for(int v=0; v<g.n; v++) sPre[v]=s[v];   // Alg. 4 lines 12-13
    }

    // ---- def-update (Alg. 5) ----
    // Cluster the (incrementally maintained) supergraph and propagate the
    // level-P labels back down to level-0 vertices.
    {
      if(Hsup.n>1){
        std::vector<int> hf(Hsup.n); for(int i=0;i<Hsup.n;i++) hf[i]=i;
        Graph hgt = Hsup;
        for(int p=1;p<P;p++){
          local_moving(hgt, hf, rng, gamma);
          int nc=renumber(hf);
          if(nc==hgt.n) break;
          std::vector<int> sub=refine(hgt, hf, nc, Instance::LEIDEN, rng, gamma);
          int S=renumber(sub);
          if(S>=hgt.n) break;
          std::vector<int> lift(S); for(int u=0;u<hgt.n;u++) lift[sub[u]]=hf[u];
          hgt = aggregate(hgt, sub, S);
          hf = lift; renumber(hf);
        }
        for(int v=0; v<g.n; v++) if(s[v] < (int)hf.size()) f[v]=hf[s[v]];
        renumber(f);
        ctot.clear(); for(int v=0;v<g.n;v++) ctot[f[v]] += g.k[v];
      }
    }

    auto t1=std::chrono::high_resolution_clock::now();
    metrics<<b<<"\t"<<std::chrono::duration<double,std::milli>(t1-t0).count()<<"\t"
           <<modularity(g,f)<<"\t"<<count_comm(f)<<"\t0\n"; metrics.flush();
    write_com(b,f);
  }
  fprintf(stderr,"[hit-leiden] done %d batches\n",nB);
  return 0;
}
