"""
Dataset Generation from Real Water Networks (EPANET .inp files).

Loads a real network topology via `wntr` and generates synthetic
leak scenarios on it, producing 7-dim enriched node features.

Key difference from synthetic pipeline: uses **pressure residuals**
(observed - no-leak baseline) instead of absolute pressures, because
real networks have huge spatial pressure variation that masks the leak signal.

Supported networks:
- BattLeDIM L-Town (785 nodes, 909 links)
- Any EPANET-compatible .inp file
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import networkx as nx
from tqdm import tqdm

from .simulator import HydraulicSimulator
from .dataset_gen import _compute_static_features, NODE_FEAT_DIM, LeakDataset
from ..utils.graph import load_network_from_inp, select_sensor_nodes


def generate_real_dataset(
    inp_path: str,
    num_samples: int,
    sensor_ratio: float = 0.25,
    demand_mean: float = 1.0,
    demand_std: float = 0.5,
    reservoir_head: float = 50.0,
    leak_probability: float = 0.8,
    leak_magnitude_min: float = 5.0,
    leak_magnitude_max: float = 50.0,
    noise_std: float = 0.1,
    seed: int = 42,
    output_path: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate a dataset using a real water network topology.

    Uses **pressure residuals** (observed - no-leak baseline) as the primary
    features instead of absolute pressures. This is critical for real networks
    where spatial pressure variation (50–3600m) dwarfs the leak signal (~4–40m).

    Args:
        inp_path: Path to the EPANET .inp file.
        num_samples: Number of samples to generate.
        sensor_ratio: Fraction of junction nodes to use as sensors.
        demand_mean: Mean demand value at junction nodes.
        demand_std: Standard deviation of demand.
        reservoir_head: Fixed head at reservoir/tank nodes (meters).
        leak_probability: Probability that a sample contains a leak.
        leak_magnitude_min: Minimum leak magnitude.
        leak_magnitude_max: Maximum leak magnitude.
        noise_std: Sensor measurement noise standard deviation.
        seed: Random seed for reproducibility.
        output_path: If set, save the dataset to this path.
        show_progress: Show progress bar.

    Returns:
        Dataset dictionary compatible with LeakDataset.
    """
    rng = np.random.RandomState(seed)

    # ── Load real network ──────────────────────────────────────────
    print(f"Loading network from: {inp_path}")
    G, reservoir_nodes, meta = load_network_from_inp(inp_path)
    num_nodes = meta["num_nodes"]
    conductances = meta["conductances"]
    positions = meta["positions"]

    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {meta['num_edges']}")
    print(f"  Reservoirs/Tanks: {reservoir_nodes}")
    print(f"  Source: {meta['source']}")

    # ── Select sensor nodes ────────────────────────────────────────
    sensor_nodes = select_sensor_nodes(
        G, sensor_ratio, exclude_nodes=reservoir_nodes, seed=seed
    )
    print(f"  Sensors: {len(sensor_nodes)} ({sensor_ratio*100:.0f}%)")

    # ── Build simulator ────────────────────────────────────────────
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=reservoir_head,
        conductances=conductances,
        seed=seed,
    )

    # ── Edge features ──────────────────────────────────────────────
    edge_index_np, edge_attr_np = simulator.get_edge_features()
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    # ── Sensor mask ────────────────────────────────────────────────
    sensor_mask = np.zeros(num_nodes, dtype=np.float32)
    sensor_mask[sensor_nodes] = 1.0
    sensor_mask_tensor = torch.tensor(sensor_mask, dtype=torch.float32)

    # ── Static features ────────────────────────────────────────────
    static_feats = _compute_static_features(
        G, num_nodes, sensor_mask, reservoir_nodes, simulator.edge_conductance
    )

    # ── Leak candidates (exclude reservoirs/tanks) ─────────────────
    leak_candidates = [n for n in range(num_nodes) if n not in reservoir_nodes]

    # ── Calibrate residual scale ───────────────────────────────────
    # Compute typical max residual from a mid-range leak to set normalization
    # This ensures Feature 0 has std ≈ 1 across samples
    cal_demand = np.full(num_nodes, demand_mean, dtype=np.float64)
    for r in reservoir_nodes:
        cal_demand[r] = 0.0
    h_baseline = simulator.solve(cal_demand)
    mid_leak_mag = (leak_magnitude_min + leak_magnitude_max) / 2.0
    sample_nodes = rng.choice(leak_candidates, size=min(20, len(leak_candidates)), replace=False)
    max_residuals = []
    for sn in sample_nodes:
        h_leak = simulator.solve(cal_demand, leak_node=int(sn), leak_magnitude=mid_leak_mag)
        max_residuals.append(np.abs(h_leak - h_baseline)[sensor_nodes].max())
    residual_scale = max(np.median(max_residuals), noise_std * 10.0)
    print(f"  Residual scale: {residual_scale:.4f} (noise_std={noise_std})")
    print(f"  Leak magnitudes: [{leak_magnitude_min}, {leak_magnitude_max}]")

    # ── Generate samples ───────────────────────────────────────────
    all_node_features = []
    all_y_node = []
    all_has_leak = []
    all_leak_nodes = []
    all_leak_magnitudes = []
    all_full_pressures = []
    all_demands = []

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating samples")

    for _ in iterator:
        # Random demand
        demand = simulator.generate_demand(mean=demand_mean, std=demand_std)

        # Leak decision
        has_leak = rng.random() < leak_probability

        if has_leak:
            leak_node = rng.choice(leak_candidates)
            leak_magnitude = rng.uniform(leak_magnitude_min, leak_magnitude_max)
        else:
            leak_node = -1
            leak_magnitude = 0.0

        # Solve WITHOUT leak (per-sample baseline for this demand)
        h_no_leak = simulator.solve(demand)

        # Solve WITH leak
        if has_leak:
            heads = simulator.solve(
                demand,
                leak_node=leak_node,
                leak_magnitude=leak_magnitude,
            )
        else:
            heads = h_no_leak

        # Noisy sensor readings
        sensor_readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std)

        # Per-node baseline at sensor locations (no noise)
        baseline_at_sensors = h_no_leak[sensor_nodes]

        # ── Build enriched 7-dim node features ─────────────────────
        node_features = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)

        # Feature 0: PRESSURE RESIDUAL (observed - baseline) / scale
        # This is the KEY fix: removes spatial variation, reveals leak anomaly
        # Without leak: ≈ noise/scale ≈ 0
        # With leak near sensor: ≈ delta_p/scale >> 0
        residuals = sensor_readings - baseline_at_sensors
        for idx, s_node in enumerate(sensor_nodes):
            node_features[s_node, 0] = residuals[idx] / residual_scale

        # Feature 1: Sensor mask
        node_features[:, 1] = sensor_mask

        # Features 2-5: Static features (degree, reservoir_dist, is_reservoir, mean_cond)
        node_features[:, 2:6] = static_feats

        # Feature 6: Residual deviation from mean sensor residual
        # Highlights which sensors see the most anomaly
        if len(residuals) > 0:
            mean_residual = residuals.mean()
            for idx, s_node in enumerate(sensor_nodes):
                node_features[s_node, 6] = (residuals[idx] - mean_residual) / residual_scale

        # Labels
        y_node = np.zeros(num_nodes, dtype=np.float32)
        if has_leak:
            y_node[leak_node] = 1.0

        all_node_features.append(torch.tensor(node_features, dtype=torch.float32))
        all_y_node.append(torch.tensor(y_node, dtype=torch.float32))
        all_has_leak.append(torch.tensor(1.0 if has_leak else 0.0, dtype=torch.float32))
        all_leak_nodes.append(torch.tensor(leak_node, dtype=torch.long))
        all_leak_magnitudes.append(torch.tensor(leak_magnitude, dtype=torch.float32))
        all_full_pressures.append(torch.tensor(heads, dtype=torch.float32))
        all_demands.append(torch.tensor(demand, dtype=torch.float32))

    # ── Stack into dataset ─────────────────────────────────────────
    dataset = {
        "node_features": torch.stack(all_node_features),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y_node": torch.stack(all_y_node),
        "has_leak": torch.stack(all_has_leak),
        "leak_nodes": torch.stack(all_leak_nodes),
        "leak_magnitudes": torch.stack(all_leak_magnitudes),
        "sensor_mask": sensor_mask_tensor,
        "reservoir_nodes": reservoir_nodes,
        "num_nodes": num_nodes,
        "positions": positions,
        "sensor_nodes": sensor_nodes,
        "full_pressures": torch.stack(all_full_pressures),
        "demands": torch.stack(all_demands),
        "conductances": torch.tensor(conductances, dtype=torch.float32),
        # Real network metadata
        "network_source": inp_path,
        "name_to_id": meta["name_to_id"],
        "id_to_name": meta["id_to_name"],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)

    return dataset

