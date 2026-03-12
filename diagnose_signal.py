"""Diagnose leak signal strength in the generated dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import networkx as nx

data = torch.load("data/real/train.pt", weights_only=False)

nf = data["node_features"]
y = data["y_node"]
has_leak = data["has_leak"]
pressures = data["full_pressures"]
sm = data["sensor_mask"]

num_nodes = nf.shape[1]
sensor_count = int(sm.sum().item())
leak_mask = has_leak > 0.5
sensor_indices = (sm > 0.5).nonzero(as_tuple=True)[0]

print("=" * 60)
print("DATASET DIAGNOSTICS")
print("=" * 60)
print(f"Nodes: {num_nodes}, Sensors: {sensor_count} ({100*sensor_count/num_nodes:.1f}%)")
print(f"Leak samples: {leak_mask.sum().item()}/{len(has_leak)}")
print()

# 1. Full pressure stats
p = pressures.numpy()
print(f"Pressure: mean={p.mean():.4f}, std={p.std():.4f}, range=[{p.min():.4f}, {p.max():.4f}]")
print()

# 2. Feature statistics
for dim, name in [(0, "Normalized pressure"), (6, "Pressure deviation")]:
    f = nf[:, sensor_indices, dim]
    print(f"Feature {dim} ({name}): mean={f.mean():.6f}, std={f.std():.6f}")
print()

# 3. Raw pressure impact of leaks
from src.utils.graph import load_network_from_inp
from src.sim.simulator import HydraulicSimulator

G, res_nodes, meta = load_network_from_inp("data/networks/L-TOWN.inp")
sim = HydraulicSimulator(G, res_nodes, 50.0, meta["conductances"], seed=42)
base_demand = sim.generate_demand(mean=1.0, std=0.5)
h0 = sim.solve(base_demand)

test_nodes = np.random.RandomState(42).choice(
    [n for n in range(num_nodes) if n not in res_nodes], size=10
)
print("--- Raw Pressure Impact of Leaks ---")
for mag in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    deltas = []
    for tn in test_nodes:
        h1 = sim.solve(base_demand, leak_node=int(tn), leak_magnitude=mag)
        deltas.append(np.abs(h1 - h0).max())
    print(f"  leak_mag={mag:6.1f} -> max |delta_p| = {np.mean(deltas):.6f}, noise_std=0.1, SNR={np.mean(deltas)/0.1:.3f}")

print()
print("--- Noise vs Signal ---")
print(f"  noise_std = 0.1")
print(f"  pressure_std = {p.std():.6f}")
print(f"  Noise / pressure_std = {0.1 / p.std():.6f}")
