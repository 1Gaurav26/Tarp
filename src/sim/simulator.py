"""
Linear Hydraulic Simulator for Water Pipe Networks.

This module implements a simplified linear hydraulic model where:
- Node head h represents pressure
- Reservoir nodes have fixed head H0
- Pipe flow Q_ij = g_e * (h_i - h_j) where g_e is conductance
- Node continuity: sum_j Q_ij = d_i (demand at node i)
- System solved via weighted Laplacian with boundary conditions
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import spsolve


class HydraulicSimulator:
    """
    Linear hydraulic simulator for water pipe networks.
    
    Solves the system of equations:
        L @ h = d
    where L is the weighted Laplacian (from pipe conductances),
    h is the vector of node heads (pressures), and d is the demand vector.
    
    Reservoir nodes have fixed head (boundary conditions).
    """
    
    def __init__(
        self,
        G: nx.Graph,
        reservoir_nodes: List[int],
        reservoir_head: float = 50.0,
        conductances: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the hydraulic simulator.
        
        Args:
            G: NetworkX graph representing the pipe network.
            reservoir_nodes: List of node indices with fixed head.
            reservoir_head: Fixed head value at reservoir nodes (meters).
            conductances: Optional array of pipe conductances. If None,
                         conductances are generated randomly.
            seed: Random seed for reproducibility.
        """
        self.G = G
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
        self.reservoir_nodes = set(reservoir_nodes)
        self.reservoir_head = reservoir_head
        self.rng = np.random.RandomState(seed)
        
        # Non-reservoir nodes (where we solve for head)
        self.free_nodes = [n for n in range(self.num_nodes) if n not in self.reservoir_nodes]
        self.free_node_set = set(self.free_nodes)
        
        # Create node index mappings
        self.free_to_global = {i: n for i, n in enumerate(self.free_nodes)}
        self.global_to_free = {n: i for i, n in enumerate(self.free_nodes)}
        
        # Set up conductances
        self.edges = list(G.edges())
        if conductances is not None:
            self.conductances = conductances
        else:
            self.conductances = self.rng.uniform(0.5, 2.0, size=len(self.edges))
        
        # Build edge-to-conductance mapping
        self.edge_conductance = {}
        for idx, (i, j) in enumerate(self.edges):
            self.edge_conductance[(i, j)] = self.conductances[idx]
            self.edge_conductance[(j, i)] = self.conductances[idx]
        
        # Build the weighted Laplacian for free nodes
        self._build_laplacian()
    
    def _build_laplacian(self) -> None:
        """Build the weighted Laplacian matrix for the linear system."""
        n_free = len(self.free_nodes)
        
        # Build sparse Laplacian for free nodes
        row_indices = []
        col_indices = []
        values = []
        
        # Also compute the constant term from reservoir nodes
        self.reservoir_contribution = np.zeros(n_free)
        
        for i, node_i in enumerate(self.free_nodes):
            diag_sum = 0.0
            
            for neighbor in self.G.neighbors(node_i):
                g = self.edge_conductance[(node_i, neighbor)]
                diag_sum += g
                
                if neighbor in self.free_node_set:
                    # Off-diagonal entry
                    j = self.global_to_free[neighbor]
                    row_indices.append(i)
                    col_indices.append(j)
                    values.append(-g)
                else:
                    # Neighbor is a reservoir - contributes to RHS
                    self.reservoir_contribution[i] += g * self.reservoir_head
            
            # Diagonal entry
            row_indices.append(i)
            col_indices.append(i)
            values.append(diag_sum)
        
        self.L_free = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_free, n_free)
        )
    
    def solve(
        self,
        demand: np.ndarray,
        leak_node: Optional[int] = None,
        leak_magnitude: float = 0.0,
    ) -> np.ndarray:
        """
        Solve for node heads given demands and optional leak.
        
        Args:
            demand: Demand vector for all nodes (positive = outflow).
            leak_node: Optional node where leak occurs.
            leak_magnitude: Magnitude of leak (added to demand at leak_node).
        
        Returns:
            Array of node heads (pressures) for all nodes.
        """
        # Adjust demand for leak
        demand_adj = demand.copy()
        if leak_node is not None and leak_magnitude > 0:
            demand_adj[leak_node] += leak_magnitude
        
        # Extract demands for free nodes
        demand_free = demand_adj[self.free_nodes]
        
        # RHS = demand + contribution from reservoir heads
        rhs = demand_free + self.reservoir_contribution
        
        # Solve the linear system
        h_free = spsolve(self.L_free, rhs)
        
        # Assemble full head vector
        h = np.zeros(self.num_nodes)
        for i, node in enumerate(self.free_nodes):
            h[node] = h_free[i]
        for node in self.reservoir_nodes:
            h[node] = self.reservoir_head
        
        return h
    
    def get_sensor_readings(
        self,
        heads: np.ndarray,
        sensor_nodes: List[int],
        noise_std: float = 0.1,
    ) -> np.ndarray:
        """
        Get noisy sensor readings from the pressure field.
        
        Args:
            heads: Full head (pressure) vector.
            sensor_nodes: List of sensor node indices.
            noise_std: Standard deviation of Gaussian noise.
        
        Returns:
            Noisy pressure readings at sensor nodes.
        """
        readings = heads[sensor_nodes].copy()
        if noise_std > 0:
            readings += self.rng.normal(0, noise_std, size=len(sensor_nodes))
        return readings
    
    def generate_demand(
        self,
        mean: float = 1.0,
        std: float = 0.5,
    ) -> np.ndarray:
        """
        Generate random demand vector.
        
        Args:
            mean: Mean demand value.
            std: Standard deviation of demand.
        
        Returns:
            Demand vector for all nodes (0 for reservoirs).
        """
        demand = self.rng.normal(mean, std, size=self.num_nodes)
        demand = np.maximum(demand, 0.1)  # Ensure positive demands
        
        # No demand at reservoir nodes
        for node in self.reservoir_nodes:
            demand[node] = 0.0
        
        return demand
    
    def get_edge_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get edge features for the graph.
        
        Returns:
            Tuple of (edge_index, edge_attr) where:
            - edge_index: [2, num_edges*2] tensor of bidirectional edges
            - edge_attr: [num_edges*2, 2] tensor of (conductance, length)
        """
        edge_list = []
        edge_attr = []
        
        for idx, (i, j) in enumerate(self.edges):
            g = self.conductances[idx]
            # Approximate length as inverse conductance (or could use actual length)
            length = 1.0 / g
            
            # Add both directions
            edge_list.append([i, j])
            edge_attr.append([g, length])
            edge_list.append([j, i])
            edge_attr.append([g, length])
        
        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_attr)
        
        return edge_index, edge_attr
    
    def estimate_leak_magnitude_physics(
        self,
        candidate_node: int,
        sensor_nodes: List[int],
        observed_pressures: np.ndarray,
        base_demand: np.ndarray,
    ) -> float:
        """
        Estimate leak magnitude at a candidate node using physics-based fitting.
        
        Uses closed-form solution by noting that the relationship between
        leak magnitude and sensor pressures is linear. We find the magnitude
        that minimizes the L2 error at sensor nodes.
        
        Args:
            candidate_node: Node where leak is hypothesized.
            sensor_nodes: List of sensor node indices.
            observed_pressures: Observed pressures at sensor nodes.
            base_demand: Base demand vector (without leak).
        
        Returns:
            Estimated leak magnitude.
        """
        # Solve with no leak
        h0 = self.solve(base_demand, leak_node=None, leak_magnitude=0.0)
        p0 = h0[sensor_nodes]
        
        # Solve with unit leak at candidate node
        h1 = self.solve(base_demand, leak_node=candidate_node, leak_magnitude=1.0)
        p1 = h1[sensor_nodes]
        
        # The pressure change per unit leak
        delta_p = p1 - p0
        
        # Observed delta from no-leak state
        delta_obs = observed_pressures - p0
        
        # Least squares: q = (delta_p^T @ delta_obs) / (delta_p^T @ delta_p)
        delta_p_norm_sq = np.dot(delta_p, delta_p)
        
        if delta_p_norm_sq < 1e-10:
            return 0.0
        
        q_est = np.dot(delta_p, delta_obs) / delta_p_norm_sq
        
        # Clamp to reasonable range
        q_est = max(0.0, q_est)
        
        return q_est
    
    def compute_sensitivity_matrix(
        self,
        sensor_nodes: List[int],
        base_demand: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sensitivity matrix: how sensor pressures change with unit leak at each node.
        
        Args:
            sensor_nodes: List of sensor node indices.
            base_demand: Base demand vector.
        
        Returns:
            Sensitivity matrix of shape [num_sensors, num_nodes].
        """
        num_sensors = len(sensor_nodes)
        
        # Baseline pressures (no leak)
        h0 = self.solve(base_demand)
        p0 = h0[sensor_nodes]
        
        sensitivity = np.zeros((num_sensors, self.num_nodes))
        
        for node in range(self.num_nodes):
            if node in self.reservoir_nodes:
                continue  # Can't have leak at reservoir
            
            h1 = self.solve(base_demand, leak_node=node, leak_magnitude=1.0)
            p1 = h1[sensor_nodes]
            sensitivity[:, node] = p1 - p0
        
        return sensitivity
