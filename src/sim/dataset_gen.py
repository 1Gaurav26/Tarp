"""
Dataset Generation for Water Leak Localization (Production).

Generates synthetic datasets with enriched 7-dim node features:
1. Measured pressure (normalized, 0 for non-sensors)
2. Sensor mask (binary)
3. Node degree (normalized)
4. Distance to nearest reservoir (normalized)
5. Is reservoir flag
6. Mean neighbor conductance (normalized)
7. Pressure deviation from global mean
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from tqdm import tqdm

from .simulator import HydraulicSimulator
from ..utils.graph import (
    create_random_graph,
    select_sensor_nodes,
    select_reservoir_nodes,
    get_node_positions,
)

# Number of node feature dimensions in the enriched format
NODE_FEAT_DIM = 7


class LeakDataset(Dataset):
    """
    PyTorch Dataset for water leak localization.
    
    Each sample contains 7-dim node features and associated labels.
    """
    
    def __init__(self, data_path: str):
        self.data = torch.load(data_path, weights_only=False)
        
        # Shared graph structure
        self.edge_index = self.data["edge_index"]
        self.edge_attr = self.data["edge_attr"]
        self.sensor_mask = self.data["sensor_mask"]
        self.reservoir_nodes = self.data["reservoir_nodes"]
        self.num_nodes = self.data["num_nodes"]
        self.positions = self.data.get("positions", None)
        
        # Per-sample data
        self.node_features = self.data["node_features"]
        self.y_node = self.data["y_node"]
        self.has_leak = self.data["has_leak"]
        self.leak_nodes = self.data["leak_nodes"]
        self.leak_magnitudes = self.data["leak_magnitudes"]
        
        self.full_pressures = self.data.get("full_pressures", None)
        self.demands = self.data.get("demands", None)
    
    def __len__(self) -> int:
        return len(self.node_features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "node_features": self.node_features[idx],
            "edge_index": self.edge_index,
            "edge_attr": self.edge_attr,
            "y_node": self.y_node[idx],
            "has_leak": self.has_leak[idx],
            "leak_node": self.leak_nodes[idx],
            "leak_magnitude": self.leak_magnitudes[idx],
            "sensor_mask": self.sensor_mask,
        }
    
    def get_graph(self) -> nx.Graph:
        """Reconstruct NetworkX graph from edge_index."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        edges = self.edge_index.numpy().T
        unique_edges = set()
        for i, j in edges:
            if (j, i) not in unique_edges:
                unique_edges.add((i, j))
        G.add_edges_from(unique_edges)
        return G


def _compute_static_features(
    G: nx.Graph,
    num_nodes: int,
    sensor_mask: np.ndarray,
    reservoir_nodes: List[int],
    edge_conductance: Dict,
) -> np.ndarray:
    """
    Compute static (per-graph, not per-sample) node features.
    
    Returns:
        static_feats: [num_nodes, 4] array of:
            - node_degree (normalized)
            - distance_to_nearest_reservoir (normalized)
            - is_reservoir (binary)
            - mean_neighbor_conductance (normalized)
    """
    static = np.zeros((num_nodes, 4), dtype=np.float32)
    
    # 1. Node degree (normalized)
    degrees = np.array([G.degree(n) for n in range(num_nodes)], dtype=np.float32)
    max_deg = degrees.max() if degrees.max() > 0 else 1.0
    static[:, 0] = degrees / max_deg
    
    # 2. Distance to nearest reservoir (normalized BFS distance)
    reservoir_set = set(reservoir_nodes)
    if len(reservoir_set) > 0:
        # Multi-source BFS from all reservoirs
        dist = np.full(num_nodes, fill_value=float('inf'), dtype=np.float32)
        for r in reservoir_nodes:
            lengths = nx.single_source_shortest_path_length(G, r)
            for node, d in lengths.items():
                dist[node] = min(dist[node], d)
        max_dist = dist[dist < float('inf')].max() if np.any(dist < float('inf')) else 1.0
        dist[dist == float('inf')] = max_dist + 1
        static[:, 1] = dist / (max_dist + 1)
    
    # 3. Is reservoir (binary)
    for r in reservoir_nodes:
        static[r, 2] = 1.0
    
    # 4. Mean neighbor conductance (normalized)
    mean_cond = np.zeros(num_nodes, dtype=np.float32)
    for n in range(num_nodes):
        neighbors = list(G.neighbors(n))
        if len(neighbors) > 0:
            conds = [edge_conductance.get((n, nb), 1.0) for nb in neighbors]
            mean_cond[n] = np.mean(conds)
    max_cond = mean_cond.max() if mean_cond.max() > 0 else 1.0
    static[:, 3] = mean_cond / max_cond
    
    return static


def generate_dataset(
    num_samples: int,
    num_nodes: int = 200,
    edge_probability: float = 0.08,
    num_reservoirs: int = 2,
    sensor_ratio: float = 0.2,
    demand_mean: float = 1.0,
    demand_std: float = 0.5,
    reservoir_head: float = 50.0,
    leak_probability: float = 0.8,
    leak_magnitude_min: float = 0.5,
    leak_magnitude_max: float = 5.0,
    noise_std: float = 0.1,
    min_conductance: float = 0.5,
    max_conductance: float = 2.0,
    seed: int = 42,
    output_path: Optional[str] = None,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate a synthetic dataset with enriched 7-dim node features.
    """
    rng = np.random.RandomState(seed)
    
    # Create graph
    G = create_random_graph(num_nodes, edge_probability, seed=seed)
    
    # Select nodes
    reservoir_nodes = select_reservoir_nodes(G, num_reservoirs, seed=seed)
    sensor_nodes = select_sensor_nodes(G, sensor_ratio, exclude_nodes=reservoir_nodes, seed=seed)
    
    # Random conductances
    num_edges = G.number_of_edges()
    conductances = rng.uniform(min_conductance, max_conductance, size=num_edges)
    
    # Simulator
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=reservoir_head,
        conductances=conductances,
        seed=seed,
    )
    
    # Edge features
    edge_index_np, edge_attr_np = simulator.get_edge_features()
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    
    # Sensor mask
    sensor_mask = np.zeros(num_nodes, dtype=np.float32)
    sensor_mask[sensor_nodes] = 1.0
    sensor_mask_tensor = torch.tensor(sensor_mask, dtype=torch.float32)
    
    # Positions for visualization
    positions = get_node_positions(G, layout="spring")
    
    # Compute static features (degree, reservoir distance, is_reservoir, mean_cond)
    static_feats = _compute_static_features(
        G, num_nodes, sensor_mask, reservoir_nodes, simulator.edge_conductance
    )
    
    # Leak candidates (exclude reservoirs)
    leak_candidates = [n for n in range(num_nodes) if n not in reservoir_nodes]
    
    # Pre-compute baseline pressure for normalization reference
    baseline_demand = np.full(num_nodes, demand_mean, dtype=np.float64)
    for r in reservoir_nodes:
        baseline_demand[r] = 0.0
    baseline_heads = simulator.solve(baseline_demand)
    pressure_mean = baseline_heads.mean()
    pressure_std = baseline_heads.std() if baseline_heads.std() > 0 else 1.0
    
    # Generate samples
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
        
        # Solve
        heads = simulator.solve(
            demand,
            leak_node=leak_node if has_leak else None,
            leak_magnitude=leak_magnitude,
        )
        
        # Noisy sensor readings
        sensor_readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std)
        
        # Build enriched 7-dim node features
        node_features = np.zeros((num_nodes, NODE_FEAT_DIM), dtype=np.float32)
        
        # Feature 0: Normalized measured pressure (0 for non-sensor nodes)
        for idx, s_node in enumerate(sensor_nodes):
            node_features[s_node, 0] = (sensor_readings[idx] - pressure_mean) / pressure_std
        
        # Feature 1: Sensor mask
        node_features[:, 1] = sensor_mask
        
        # Features 2-5: Static features (degree, reservoir_dist, is_reservoir, mean_cond)
        node_features[:, 2:6] = static_feats
        
        # Feature 6: Pressure deviation from global sensor mean
        if len(sensor_readings) > 0:
            global_sensor_mean = sensor_readings.mean()
            for idx, s_node in enumerate(sensor_nodes):
                node_features[s_node, 6] = (sensor_readings[idx] - global_sensor_mean) / pressure_std
        
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
    
    # Stack
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
    }
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, output_path)
    
    return dataset


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a saved dataset."""
    dataset = LeakDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching graph data."""
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": batch[0]["edge_index"],
        "edge_attr": batch[0]["edge_attr"],
        "y_node": torch.stack([b["y_node"] for b in batch]),
        "has_leak": torch.stack([b["has_leak"] for b in batch]),
        "leak_node": torch.stack([b["leak_node"] for b in batch]),
        "leak_magnitude": torch.stack([b["leak_magnitude"] for b in batch]),
        "sensor_mask": batch[0]["sensor_mask"],
    }
