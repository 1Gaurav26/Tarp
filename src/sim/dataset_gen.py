"""
Dataset Generation for Water Leak Localization.

Generates synthetic datasets with:
- Random demand patterns
- Optional leak injection at nodes
- Noisy sensor observations
- Labels for leak node and magnitude
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


class LeakDataset(Dataset):
    """
    PyTorch Dataset for water leak localization.
    
    Each sample contains:
    - node_features: [num_nodes, feat_dim] with (pressure_or_0, sensor_mask, ...)
    - edge_index: [2, num_edges*2] bidirectional edge indices
    - edge_attr: [num_edges*2, 2] edge features (conductance, length)
    - y_node: [num_nodes] one-hot leak node (all zeros if no leak)
    - has_leak: scalar (1 if leak, 0 if no leak)
    - leak_node: scalar index of leak node (-1 if no leak)
    - leak_magnitude: scalar magnitude of leak (0 if no leak)
    - sensor_mask: [num_nodes] binary mask of sensor nodes
    """
    
    def __init__(
        self,
        data_path: str,
    ):
        """
        Load dataset from file.
        
        Args:
            data_path: Path to .pt file containing the dataset.
        """
        self.data = torch.load(data_path, weights_only=False)
        
        # Store graph structure (shared across samples)
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
        
        # Optional: store full pressures for debugging
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
        # Only add unique edges (edge_index has bidirectional)
        unique_edges = set()
        for i, j in edges:
            if (j, i) not in unique_edges:
                unique_edges.add((i, j))
        
        G.add_edges_from(unique_edges)
        return G


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
    Generate a synthetic dataset for water leak localization.
    
    Args:
        num_samples: Number of samples to generate.
        num_nodes: Number of nodes in the graph.
        edge_probability: Probability of edge creation.
        num_reservoirs: Number of reservoir nodes.
        sensor_ratio: Fraction of nodes that are sensors.
        demand_mean: Mean demand value.
        demand_std: Standard deviation of demand.
        reservoir_head: Fixed head at reservoirs.
        leak_probability: Probability of a sample having a leak.
        leak_magnitude_min: Minimum leak magnitude.
        leak_magnitude_max: Maximum leak magnitude.
        noise_std: Sensor noise standard deviation.
        min_conductance: Minimum pipe conductance.
        max_conductance: Maximum pipe conductance.
        seed: Random seed.
        output_path: Optional path to save the dataset.
        show_progress: Whether to show progress bar.
    
    Returns:
        Dictionary containing the dataset.
    """
    rng = np.random.RandomState(seed)
    
    # Create graph
    G = create_random_graph(num_nodes, edge_probability, seed=seed)
    
    # Select reservoir and sensor nodes
    reservoir_nodes = select_reservoir_nodes(G, num_reservoirs, seed=seed)
    sensor_nodes = select_sensor_nodes(G, sensor_ratio, exclude_nodes=reservoir_nodes, seed=seed)
    
    # Generate random conductances
    num_edges = G.number_of_edges()
    conductances = rng.uniform(min_conductance, max_conductance, size=num_edges)
    
    # Initialize simulator
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=reservoir_head,
        conductances=conductances,
        seed=seed,
    )
    
    # Get edge features
    edge_index_np, edge_attr_np = simulator.get_edge_features()
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)
    
    # Create sensor mask
    sensor_mask = np.zeros(num_nodes, dtype=np.float32)
    sensor_mask[sensor_nodes] = 1.0
    sensor_mask_tensor = torch.tensor(sensor_mask, dtype=torch.float32)
    
    # Node positions for visualization
    positions = get_node_positions(G, layout="spring")
    
    # Candidate nodes for leaks (exclude reservoirs)
    leak_candidates = [n for n in range(num_nodes) if n not in reservoir_nodes]
    
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
        # Generate random demand
        demand = simulator.generate_demand(mean=demand_mean, std=demand_std)
        
        # Decide if this sample has a leak
        has_leak = rng.random() < leak_probability
        
        if has_leak:
            # Select random leak node
            leak_node = rng.choice(leak_candidates)
            leak_magnitude = rng.uniform(leak_magnitude_min, leak_magnitude_max)
        else:
            leak_node = -1
            leak_magnitude = 0.0
        
        # Solve for pressures
        heads = simulator.solve(
            demand,
            leak_node=leak_node if has_leak else None,
            leak_magnitude=leak_magnitude,
        )
        
        # Get noisy sensor readings
        sensor_readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std)
        
        # Build node features
        # Feature: [measured_pressure_or_0, sensor_mask]
        node_features = np.zeros((num_nodes, 2), dtype=np.float32)
        node_features[:, 1] = sensor_mask  # Sensor mask as feature
        
        for idx, s_node in enumerate(sensor_nodes):
            node_features[s_node, 0] = sensor_readings[idx]
        
        # Build labels
        y_node = np.zeros(num_nodes, dtype=np.float32)
        if has_leak:
            y_node[leak_node] = 1.0
        
        # Store
        all_node_features.append(torch.tensor(node_features, dtype=torch.float32))
        all_y_node.append(torch.tensor(y_node, dtype=torch.float32))
        all_has_leak.append(torch.tensor(1.0 if has_leak else 0.0, dtype=torch.float32))
        all_leak_nodes.append(torch.tensor(leak_node, dtype=torch.long))
        all_leak_magnitudes.append(torch.tensor(leak_magnitude, dtype=torch.float32))
        all_full_pressures.append(torch.tensor(heads, dtype=torch.float32))
        all_demands.append(torch.tensor(demand, dtype=torch.float32))
    
    # Stack all samples
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
    """
    Create a DataLoader from a saved dataset.
    
    Args:
        data_path: Path to .pt file.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
    
    Returns:
        PyTorch DataLoader.
    """
    dataset = LeakDataset(data_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching graph data.
    
    Since all samples share the same graph structure, we just stack
    the node features and labels.
    """
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": batch[0]["edge_index"],  # Same for all samples
        "edge_attr": batch[0]["edge_attr"],
        "y_node": torch.stack([b["y_node"] for b in batch]),
        "has_leak": torch.stack([b["has_leak"] for b in batch]),
        "leak_node": torch.stack([b["leak_node"] for b in batch]),
        "leak_magnitude": torch.stack([b["leak_magnitude"] for b in batch]),
        "sensor_mask": batch[0]["sensor_mask"],
    }
