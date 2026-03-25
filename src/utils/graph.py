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


def load_network_from_inp(
    inp_path: str,
) -> Tuple[nx.Graph, List[int], Dict]:
    """
    Load a water distribution network from an EPANET .inp file.

    Uses the `wntr` library to parse the .inp file and extracts:
    - NetworkX graph with integer node IDs
    - Reservoir and tank node indices
    - Pipe conductances derived from Hazen-Williams coefficients
    - Node positions from EPANET coordinates (if available)

    Args:
        inp_path: Path to the .inp file.

    Returns:
        Tuple of (G, reservoir_nodes, metadata) where:
        - G: NetworkX Graph with integer node IDs
        - reservoir_nodes: List of reservoir/tank node integer indices
        - metadata: Dict with keys:
            - 'conductances': np.array of per-edge conductance values
            - 'name_to_id': Dict mapping original node names to integer IDs
            - 'id_to_name': Dict mapping integer IDs to original node names
            - 'positions': Dict mapping integer IDs to (x, y) tuples
            - 'num_nodes': int
            - 'num_edges': int
    """
    try:
        import wntr
    except ImportError:
        raise ImportError(
            "wntr is required for loading .inp files. "
            "Install it with: pip install wntr"
        )

    # Load the EPANET model
    wn = wntr.network.WaterNetworkModel(inp_path)

    # Build node name -> integer ID mapping
    all_node_names = wn.node_name_list
    name_to_id = {name: i for i, name in enumerate(all_node_names)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    # Identify reservoir + tank nodes (both act as fixed-head boundaries)
    reservoir_names = wn.reservoir_name_list + wn.tank_name_list
    reservoir_nodes = sorted([name_to_id[n] for n in reservoir_names])

    # Build the graph with integer node IDs
    # IMPORTANT: Include ALL link types (pipes, valves, pumps) — not just pipes.
    # In many real networks (incl. L-Town), reservoirs connect to junctions
    # through valves or pumps. Skipping these creates disconnected components
    # that exclude reservoirs, making the Laplacian singular.
    num_nodes = len(all_node_names)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    raw_conductances = []

    # 1. Pipes — compute conductance from physical properties
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        start_id = name_to_id[pipe.start_node_name]
        end_id = name_to_id[pipe.end_node_name]

        diameter = pipe.diameter  # meters
        length = max(pipe.length, 1.0)  # meters (avoid div by 0)
        roughness = pipe.roughness  # Hazen-Williams C coefficient

        g = roughness * (diameter ** 2) / length
        raw_conductances.append(g)
        G.add_edge(start_id, end_id, raw_g=g)

    # 2. Valves — treat as high-conductance connections (open valves)
    for valve_name in wn.valve_name_list:
        valve = wn.get_link(valve_name)
        start_id = name_to_id[valve.start_node_name]
        end_id = name_to_id[valve.end_node_name]
        if not G.has_edge(start_id, end_id):
            raw_conductances.append(None)  # placeholder, will be set to high value
            G.add_edge(start_id, end_id, raw_g=None)

    # 3. Pumps — treat as high-conductance connections
    for pump_name in wn.pump_name_list:
        pump = wn.get_link(pump_name)
        start_id = name_to_id[pump.start_node_name]
        end_id = name_to_id[pump.end_node_name]
        if not G.has_edge(start_id, end_id):
            raw_conductances.append(None)  # placeholder
            G.add_edge(start_id, end_id, raw_g=None)

    # Replace None placeholders (valves/pumps) with max pipe conductance
    pipe_conductances = [g for g in raw_conductances if g is not None]
    if pipe_conductances:
        max_pipe_g = max(pipe_conductances)
    else:
        max_pipe_g = 1.0
        
    # Store normalized conductances directly in the graph edges
    # First apply min-max scaling to [0.5, 2.0] for non-None, and make None = 2.0 * max_pipe_g scaled
    if len(pipe_conductances) > 0 and max(pipe_conductances) > min(pipe_conductances):
        c_min, c_max = min(pipe_conductances), max(pipe_conductances)
    else:
        c_min, c_max = 0.0, 1.0
        
    for u, v, data in G.edges(data=True):
        raw_g = data.get('raw_g')
        if raw_g is None:
            g = max_pipe_g * 2.0
        else:
            g = raw_g
            
        if c_max > c_min:
            norm_g = 0.5 + 1.5 * (g - c_min) / (c_max - c_min)
        else:
            norm_g = 1.0
            
        G[u][v]['conductance'] = norm_g

    # Extract node positions from EPANET coordinates
    positions = {}
    for name, node_id in name_to_id.items():
        node = wn.get_node(name)
        coords = node.coordinates
        if coords and coords != (0, 0):
            positions[node_id] = (float(coords[0]), float(coords[1]))
        else:
            positions[node_id] = (0.0, 0.0)

    # Normalize positions to [0, 1] range for visualization
    if positions:
        xs = [p[0] for p in positions.values()]
        ys = [p[1] for p in positions.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        positions = {
            nid: ((x - x_min) / x_range, (y - y_min) / y_range)
            for nid, (x, y) in positions.items()
        }

    # Ensure graph is connected — prefer the component containing reservoirs
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Find the component that contains reservoir/tank nodes
        res_set = set(reservoir_nodes)
        best_cc = None
        for cc in components:
            if cc & res_set:
                if best_cc is None or len(cc) > len(best_cc):
                    best_cc = cc
        # Fallback to largest component if no reservoirs found
        if best_cc is None:
            best_cc = max(components, key=len)

        if len(best_cc) < num_nodes:
            print(
                f"  Warning: Network has {len(components)} components. "
                f"Using reservoir-connected component "
                f"({len(best_cc)} of {num_nodes} nodes)."
            )
            # Remap to contiguous IDs
            old_to_new = {old: new for new, old in enumerate(sorted(best_cc))}
            G_new = nx.Graph()
            G_new.add_nodes_from(range(len(best_cc)))

            for idx, (u, v) in enumerate(list(G.edges())):
                if u in best_cc and v in best_cc:
                    G_new.add_edge(old_to_new[u], old_to_new[v], conductance=G[u][v]['conductance'])

            reservoir_nodes = sorted([
                old_to_new[r] for r in reservoir_nodes if r in best_cc
            ])
            positions = {
                old_to_new[nid]: pos for nid, pos in positions.items()
                if nid in best_cc
            }
            new_name_to_id = {}
            new_id_to_name = {}
            for name, old_id in name_to_id.items():
                if old_id in best_cc:
                    new_id = old_to_new[old_id]
                    new_name_to_id[name] = new_id
                    new_id_to_name[new_id] = name
            name_to_id = new_name_to_id
            id_to_name = new_id_to_name
            G = G_new
            num_nodes = G.number_of_nodes()

    # Now we can safely build the parallel contiguous array by iterating the edges once!
    conductances = np.array([G[u][v]['conductance'] for u, v in G.edges()], dtype=np.float64)

    metadata = {
        "conductances": conductances,
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "positions": positions,
        "num_nodes": num_nodes,
        "num_edges": G.number_of_edges(),
        "source": inp_path,
    }

    return G, reservoir_nodes, metadata
