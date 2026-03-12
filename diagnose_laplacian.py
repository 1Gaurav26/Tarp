"""Investigate raw conductance values and Laplacian conditioning."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import spsolve

from src.utils.graph import load_network_from_inp

G, res_nodes, meta = load_network_from_inp("data/networks/L-TOWN.inp")
cond = meta["conductances"]

print("=" * 60)
print("CONDUCTANCE & LAPLACIAN DIAGNOSTICS")
print("=" * 60)
print(f"Nodes: {meta['num_nodes']}, Edges: {meta['num_edges']}")
print(f"Reservoirs: {res_nodes}")
print(f"Connected: {nx.is_connected(G)}")
print()

print("--- Conductances (after normalization) ---")
print(f"  min={cond.min():.6f}, max={cond.max():.6f}, mean={cond.mean():.6f}")
print(f"  ratio max/min = {cond.max()/cond.min():.2f}")
print()

# Check reservoir connectivity
print("--- Reservoir Connectivity ---")
for r in res_nodes:
    deg = G.degree(r)
    neighbors = list(G.neighbors(r))[:5]
    print(f"  Reservoir {r}: degree={deg}, neighbors={neighbors}")

# Check how many free nodes are connected to at least one reservoir
free_nodes = [n for n in range(meta['num_nodes']) if n not in res_nodes]
connected_to_res = set()
for r in res_nodes:
    for n in nx.single_source_shortest_path_length(G, r):
        connected_to_res.add(n)
print(f"  Free nodes connected to any reservoir: {len(connected_to_res) - len(res_nodes)}/{len(free_nodes)}")
print()

# Build the Laplacian and check conditioning
num_nodes = meta['num_nodes']
edges = list(G.edges())
free_set = set(free_nodes)
free_to_idx = {n: i for i, n in enumerate(free_nodes)}

# Build edge conductance map
edge_cond = {}
for idx, (u, v) in enumerate(edges):
    edge_cond[(u, v)] = cond[idx]
    edge_cond[(v, u)] = cond[idx]

n_free = len(free_nodes)
row, col, vals = [], [], []
res_contrib = np.zeros(n_free)

for i, ni in enumerate(free_nodes):
    diag = 0.0
    for nb in G.neighbors(ni):
        g = edge_cond[(ni, nb)]
        diag += g
        if nb in free_set:
            j = free_to_idx[nb]
            row.append(i)
            col.append(j)
            vals.append(-g)
        else:
            res_contrib[i] += g * 50.0
    row.append(i)
    col.append(i)
    vals.append(diag)

L = sparse.csr_matrix((vals, (row, col)), shape=(n_free, n_free))

print("--- Laplacian Matrix ---")
print(f"  Shape: {L.shape}")
print(f"  nnz: {L.nnz}")

# Check diagonal values
diag_vals = L.diagonal()
print(f"  Diagonal: min={diag_vals.min():.6f}, max={diag_vals.max():.6f}")
print(f"  Very small diag (<0.01): {(diag_vals < 0.01).sum()}")
print(f"  Zero diag: {(diag_vals == 0).sum()}")
print()

# Check isolated nodes (small diagonal = barely connected)
small_diag_nodes = np.where(diag_vals < 0.1)[0]
if len(small_diag_nodes) > 0:
    print(f"  Weakly connected free nodes (diag < 0.1): {len(small_diag_nodes)}")
    for sn in small_diag_nodes[:5]:
        real_node = free_nodes[sn]
        print(f"    Node {real_node}: diag={diag_vals[sn]:.6f}, degree={G.degree(real_node)}")

# Test solve
demand = np.ones(n_free) * 1.0
rhs = demand + res_contrib
print("\n--- Test Solve ---")
print(f"  RHS: min={rhs.min():.4f}, max={rhs.max():.4f}")

h = spsolve(L, rhs)
print(f"  Solution h: min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}")

if np.any(np.abs(h) > 1e6):
    print("  *** SOLUTION BLOWS UP! ***")
    bad = np.where(np.abs(h) > 1e6)[0]
    print(f"  {len(bad)} nodes have |h| > 1e6")
    for b in bad[:5]:
        real_node = free_nodes[b]
        print(f"    Node {real_node}: h={h[b]:.2e}, diag={diag_vals[b]:.6f}, degree={G.degree(real_node)}")
