"""
Graph utilities for NetworkX operations.

Provides functions for graph generation, conversion, and analysis.
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


def create_random_graph(
    num_nodes: int,
    edge_probability: float = 0.08,
    seed: Optional[int] = None,
    ensure_connected: bool = True,
) -> nx.Graph:
    """
    Create a random graph with Erdos-Renyi model.

    Args:
        num_nodes: Number of nodes in the graph.
        edge_probability: Probability of edge creation between nodes.
        seed: Random seed for reproducibility.
        ensure_connected: If True, ensures the graph is connected.

    Returns:
        NetworkX Graph object.
    """
    rng = np.random.RandomState(seed)
    
    while True:
        G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=rng.randint(0, 2**31))
        
        if not ensure_connected:
            break
            
        if nx.is_connected(G):
            break
        
        # Increase probability if not connected
        edge_probability = min(edge_probability * 1.1, 0.5)
    
    return G


def create_grid_graph(
    rows: int,
    cols: int,
    add_diagonals: bool = False,
) -> nx.Graph:
    """
    Create a grid-like graph (common in water networks).

    Args:
        rows: Number of rows.
        cols: Number of columns.
        add_diagonals: Whether to add diagonal edges.

    Returns:
        NetworkX Graph object.
    """
    G = nx.grid_2d_graph(rows, cols)
    
    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    if add_diagonals:
        for i in range(rows - 1):
            for j in range(cols - 1):
                node = i * cols + j
                # Add diagonal edges
                G.add_edge(node, node + cols + 1)
                G.add_edge(node + 1, node + cols)
    
    return G


def get_edge_index(G: nx.Graph) -> torch.Tensor:
    """
    Convert NetworkX graph to PyTorch edge_index format.

    Args:
        G: NetworkX Graph.

    Returns:
        Edge index tensor of shape [2, num_edges * 2] (bidirectional).
    """
    edges = list(G.edges())
    
    # Create bidirectional edges
    edge_list = []
    for i, j in edges:
        edge_list.append([i, j])
        edge_list.append([j, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Get adjacency matrix from NetworkX graph.

    Args:
        G: NetworkX Graph.

    Returns:
        Adjacency matrix as numpy array.
    """
    return nx.adjacency_matrix(G).toarray()


def compute_shortest_paths(G: nx.Graph) -> Dict[int, Dict[int, int]]:
    """
    Compute all-pairs shortest path lengths.

    Args:
        G: NetworkX Graph.

    Returns:
        Dictionary mapping node pairs to shortest path lengths.
    """
    return dict(nx.all_pairs_shortest_path_length(G))


def compute_graph_distance(G: nx.Graph, node1: int, node2: int) -> int:
    """
    Compute shortest path distance between two nodes.

    Args:
        G: NetworkX Graph.
        node1: First node.
        node2: Second node.

    Returns:
        Shortest path length (or -1 if not connected).
    """
    try:
        return nx.shortest_path_length(G, node1, node2)
    except nx.NetworkXNoPath:
        return -1


def select_sensor_nodes(
    G: nx.Graph,
    ratio: float,
    exclude_nodes: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Select sensor nodes from the graph.

    Args:
        G: NetworkX Graph.
        ratio: Fraction of nodes to select as sensors.
        exclude_nodes: Nodes to exclude from selection (e.g., reservoirs).
        seed: Random seed.

    Returns:
        List of sensor node indices.
    """
    rng = np.random.RandomState(seed)
    
    all_nodes = list(G.nodes())
    if exclude_nodes:
        all_nodes = [n for n in all_nodes if n not in exclude_nodes]
    
    num_sensors = max(1, int(len(all_nodes) * ratio))
    sensor_nodes = rng.choice(all_nodes, size=num_sensors, replace=False).tolist()
    
    return sorted(sensor_nodes)


def select_reservoir_nodes(
    G: nx.Graph,
    num_reservoirs: int,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Select reservoir nodes (typically high-degree nodes at periphery).

    Args:
        G: NetworkX Graph.
        num_reservoirs: Number of reservoir nodes.
        seed: Random seed.

    Returns:
        List of reservoir node indices.
    """
    rng = np.random.RandomState(seed)
    
    # Select nodes with lower centrality (more peripheral)
    # This mimics real water networks where reservoirs are at edges
    degrees = dict(G.degree())
    nodes_by_degree = sorted(degrees.keys(), key=lambda x: degrees[x])
    
    # Select from lower-degree nodes
    candidates = nodes_by_degree[:max(len(nodes_by_degree) // 3, num_reservoirs + 1)]
    reservoir_nodes = rng.choice(candidates, size=num_reservoirs, replace=False).tolist()
    
    return sorted(reservoir_nodes)


def get_node_positions(G: nx.Graph, layout: str = "spring") -> Dict[int, Tuple[float, float]]:
    """
    Get node positions for visualization.

    Args:
        G: NetworkX Graph.
        layout: Layout algorithm ("spring", "kamada_kawai", "spectral").

    Returns:
        Dictionary mapping nodes to (x, y) positions.
    """
    if layout == "spring":
        return nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        return nx.spectral_layout(G)
    else:
        return nx.spring_layout(G, seed=42)


def extract_subgraph(
    G: nx.Graph,
    center_node: int,
    radius: int,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Extract r-hop ego subgraph around a node.

    Args:
        G: NetworkX Graph.
        center_node: Center node for subgraph.
        radius: Radius (number of hops).

    Returns:
        Tuple of (subgraph, node_mapping) where node_mapping maps
        original node IDs to subgraph node IDs.
    """
    # Get nodes within radius hops
    nodes_in_radius = nx.single_source_shortest_path_length(G, center_node, cutoff=radius)
    subgraph_nodes = list(nodes_in_radius.keys())
    
    # Extract subgraph
    subgraph = G.subgraph(subgraph_nodes).copy()
    
    # Create mapping
    node_mapping = {old: new for new, old in enumerate(subgraph_nodes)}
    subgraph = nx.relabel_nodes(subgraph, node_mapping)
    
    return subgraph, node_mapping
