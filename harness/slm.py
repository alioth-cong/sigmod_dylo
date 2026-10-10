#!/usr/bin/env python3
"""
Genuine Smart Local Moving (SLM) — Waltman & van Eck, EPJ B 2013 — for modularity.

STATUS (2026-05-23): EXPERIMENTAL, NOT wired as the `slm-static` baseline.
Validated optimal on Zachary (Q=0.4198), but it under-merges on larger graphs
(facebook Q=0.818 / 51 communities vs Louvain 0.836 / 18) — a correct SLM is
>= Louvain, so there is a residual bug in the reduced-network merging that was
not resolved. We do NOT ship below-Louvain numbers under the SLM name. The
authoritative reference (SLM4J / Waltman-van Eck Modularity Optimizer, algorithm 3)
is the intended source but adding it as a submodule is blocked by the security
classifier; see REPLY-writing-agent P4. Kept here for the record / future fix.

This is a from-scratch implementation (no external binary), distinct from plain
Louvain: after the local-moving phase it REFINES each community by running local
moving *within* the community's induced subnetwork (from singletons), then builds a
reduced network whose nodes are those sub-communities but INITIALIZED to the original
communities, and solves it recursively. That within-community refinement is the
"smart" part — it lets a community split during aggregation and escape the Louvain
local optimum.

API: slm_clustering(g) -> membership list (over g's vertices). g is an igraph.Graph
(undirected, unweighted is fine; aggregated levels carry weights internally).

Reference structure follows the Modularity Optimizer's runSmartLocalMovingAlgorithm.
"""
from __future__ import annotations
import random


class Net:
    """Weighted undirected network for local moving.
       neigh[i] = list of (j, w); self[i] = self-loop weight; k[i] = total incident
       weight (self-loop counted twice); m2 = sum_i k[i] = 2m."""
    __slots__ = ("n", "neigh", "self", "k", "m2")

    def __init__(self, n):
        self.n = n
        self.neigh = [[] for _ in range(n)]
        self.self = [0.0] * n
        self.k = [0.0] * n
        self.m2 = 0.0

    @staticmethod
    def from_edges(n, edges):
        net = Net(n)
        adj = [dict() for _ in range(n)]
        for u, v in edges:
            if u == v:
                net.self[u] += 1.0
                net.k[u] += 2.0
                net.m2 += 2.0
                continue
            adj[u][v] = adj[u].get(v, 0.0) + 1.0
            adj[v][u] = adj[v].get(u, 0.0) + 1.0
            net.k[u] += 1.0; net.k[v] += 1.0
            net.m2 += 2.0
        for i in range(n):
            net.neigh[i] = list(adj[i].items())
        return net


def local_moving(net: Net, comm, rng, gamma=1.0):
    """One full local-moving optimisation (sweep until no node moves).
       comm: list community-id per node (mutated). Returns True if anything moved."""
    n = net.n
    if n == 1:
        return False
    m2 = net.m2
    if m2 == 0:
        return False
    # community total incident weight
    ctot = {}
    for i in range(n):
        ctot[comm[i]] = ctot.get(comm[i], 0.0) + net.k[i]
    order = list(range(n))
    moved_any = False
    improved = True
    while improved:
        improved = False
        rng.shuffle(order)
        for i in order:
            ci = comm[i]
            ki = net.k[i]
            # weights from i to each neighbouring community
            kin = {}
            for j, w in net.neigh[i]:
                cj = comm[j]
                kin[cj] = kin.get(cj, 0.0) + w
            # remove i from ci
            ctot[ci] -= ki
            kin_ci = kin.get(ci, 0.0)
            # best target: maximise  kin_C - gamma*ctot_C*ki/m2
            best_c = ci
            best_gain = kin_ci - gamma * ctot[ci] * ki / m2   # gain of staying (re-adding to ci)
            for c, kic in kin.items():
                if c == ci:
                    continue
                gain = kic - gamma * ctot.get(c, 0.0) * ki / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = c
            ctot[best_c] = ctot.get(best_c, 0.0) + ki
            if best_c != ci:
                comm[i] = best_c
                moved_any = True
                improved = True
    return moved_any


def _renumber(comm):
    remap = {}
    out = [0] * len(comm)
    for i, c in enumerate(comm):
        if c not in remap:
            remap[c] = len(remap)
        out[i] = remap[c]
    return out, len(remap)


def _reduced_net(net: Net, sub):
    """Aggregate net by sub-community labels `sub` (0..S-1) -> reduced Net."""
    S = max(sub) + 1
    r = Net(S)
    adj = [dict() for _ in range(S)]
    for i in range(net.n):
        si = sub[i]
        r.self[si] += net.self[i]
        for j, w in net.neigh[i]:
            sj = sub[j]
            if si == sj:
                r.self[si] += w / 2.0   # each intra edge seen twice over i,j
            else:
                adj[si][sj] = adj[si].get(sj, 0.0) + w
    for s in range(S):
        r.neigh[s] = list(adj[s].items())
        r.k[s] = sum(w for _, w in r.neigh[s]) + 2.0 * r.self[s]
    r.m2 = net.m2
    return r


def _refine(net: Net, comm, nclus, rng, gamma):
    """SLM refinement: within each community run local moving from singletons.
       Returns `sub` (node -> global sub-community id, 0..S-1)."""
    members = [[] for _ in range(nclus)]
    for i, c in enumerate(comm):
        members[c].append(i)
    sub = [0] * net.n
    next_id = 0
    for c in range(nclus):
        nodes = members[c]
        if len(nodes) == 1:
            sub[nodes[0]] = next_id; next_id += 1
            continue
        idx = {v: t for t, v in enumerate(nodes)}
        snet = Net(len(nodes))
        adj = [dict() for _ in range(len(nodes))]
        for v in nodes:
            snet.self[idx[v]] += net.self[v]
            for j, w in net.neigh[v]:
                if comm[j] == c and j != v:
                    a = idx[v]; b = idx[j]
                    adj[a][b] = adj[a].get(b, 0.0) + w
        for t in range(len(nodes)):
            snet.neigh[t] = list(adj[t].items())
            snet.k[t] = sum(w for _, w in snet.neigh[t]) + 2.0 * snet.self[t]
            snet.m2 += snet.k[t]
        scomm = list(range(len(nodes)))
        local_moving(snet, scomm, rng, gamma)
        scomm, ns = _renumber(scomm)
        for v in nodes:
            sub[v] = next_id + scomm[idx[v]]
        next_id += ns
    return sub, next_id


def slm_membership(net0: Net, rng, gamma=1.0, init=None):
    """One SLM pass (Modularity-Optimizer recursion unrolled into a strictly-shrinking
       aggregation loop -> always terminates). `init` = starting membership over
       net0's nodes (default singletons). Returns membership over net0's nodes."""
    net = net0
    comm = list(init) if init is not None else list(range(net.n))
    levels = []   # each: sub mapping (this-level node -> next-level/reduced node)
    while True:
        local_moving(net, comm, rng, gamma)
        comm, nclus = _renumber(comm)
        if nclus == net.n:
            break                                  # nothing merged -> converged
        sub, S = _refine(net, comm, nclus, rng, gamma)
        if S == net.n:
            break                                  # refinement didn't reduce -> stop, use comm
        rnet = _reduced_net(net, sub)
        rcomm = [0] * S                            # reduced net init = ORIGINAL communities
        for i in range(net.n):
            rcomm[sub[i]] = comm[i]
        levels.append(sub)
        net, comm = rnet, rcomm
    memb = comm                                    # over top-level nodes
    for sub in reversed(levels):                   # lift down to net0 vertices
        memb = [memb[sub[i]] for i in range(len(sub))]
    return memb


def slm_clustering(g, seed=42, n_random_starts=1, gamma=1.0):
    """Run SLM on an igraph.Graph; return membership (over g's vertices)."""
    edges = g.get_edgelist()
    n = g.vcount()
    best_comm = None; best_q = None
    for s in range(n_random_starts):
        rng = random.Random(seed + s)
        net = Net.from_edges(n, edges)
        # Iterative variant (Waltman & van Eck 2013, sec. 2.4): re-run SLM starting
        # from the previous output until modularity no longer improves -> locally
        # optimal w.r.t. community merging AND individual node movements.
        comm = None; prev_q = float("-inf")
        for _ in range(50):
            comm = slm_membership(net, rng, gamma, init=comm)
            q = modularity(g, comm)
            if q <= prev_q + 1e-10:
                break
            prev_q = q
        if best_q is None or prev_q > best_q:
            best_q, best_comm = prev_q, comm
    return best_comm


def modularity(g, membership, gamma=1.0):
    return g.modularity(membership, resolution=gamma)


if __name__ == "__main__":
    import igraph as ig
    for name in ["Zachary"]:
        g = ig.Graph.Famous(name)
        m = slm_clustering(g, seed=42, n_random_starts=10)
        print(f"{name}: SLM Q={g.modularity(m):.4f} comms={len(set(m))}  "
              f"(Louvain {g.community_multilevel().modularity:.4f})")
