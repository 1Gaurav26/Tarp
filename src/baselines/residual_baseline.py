"""
Residual Ranking Baseline for Water Leak Localization.

This baseline ranks nodes based on pressure residual scores computed
by comparing observed pressures to expected (no-leak) pressures.
"""

from typing import List, Optional, Tuple

import numpy as np
import networkx as nx

from ..sim.simulator import HydraulicSimulator


class ResidualRankingBaseline:
    """
    Baseline that ranks leak candidates by pressure residual scores.
    
    For each candidate node, estimates what the sensor pressures would be
    if there were a unit leak at that node, and computes the correlation
    with the observed residual pattern.
    """
    
    def __init__(
        self,
        simulator: HydraulicSimulator,
        sensor_nodes: List[int],
    ):
        """
        Initialize the baseline.
        
        Args:
            simulator: Hydraulic simulator.
            sensor_nodes: List of sensor node indices.
        """
        self.simulator = simulator
        self.sensor_nodes = sensor_nodes
        self.num_nodes = simulator.num_nodes
        self.reservoir_nodes = simulator.reservoir_nodes
        
        # Precompute sensitivity matrix
        self.sensitivity = None
    
    def compute_sensitivity(self, base_demand: np.ndarray) -> None:
        """
        Precompute sensitivity matrix for the given base demand.
        
        Args:
            base_demand: Base demand vector.
        """
        self.sensitivity = self.simulator.compute_sensitivity_matrix(
            self.sensor_nodes, base_demand
        )
        self.base_demand = base_demand
        self.baseline_heads = self.simulator.solve(base_demand)
        self.baseline_pressures = self.baseline_heads[self.sensor_nodes]
    
    def compute_scores(
        self,
        observed_pressures: np.ndarray,
        base_demand: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute leak probability scores for all nodes.
        
        Uses correlation between observed pressure residual and
        expected residual pattern for each candidate.
        
        Args:
            observed_pressures: Observed pressures at sensor nodes.
            base_demand: Base demand (if different from precomputed).
        
        Returns:
            Array of scores for each node (higher = more likely leak).
        """
        if base_demand is not None and self.sensitivity is None:
            self.compute_sensitivity(base_demand)
        elif self.sensitivity is None:
            raise ValueError("Must call compute_sensitivity first or provide base_demand")
        
        # Observed residual
        residual_obs = observed_pressures - self.baseline_pressures
        
        # Compute correlation with each candidate's sensitivity pattern
        scores = np.zeros(self.num_nodes)
        
        for node in range(self.num_nodes):
            if node in self.reservoir_nodes:
                scores[node] = -np.inf  # Can't have leak at reservoir
                continue
            
            sensitivity_pattern = self.sensitivity[:, node]
            
            # Compute correlation score
            if np.std(sensitivity_pattern) > 1e-8 and np.std(residual_obs) > 1e-8:
                correlation = np.corrcoef(sensitivity_pattern, residual_obs)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            
            # Also use L2 distance (lower is better, so negate)
            # But first estimate magnitude
            sens_norm = np.dot(sensitivity_pattern, sensitivity_pattern)
            if sens_norm > 1e-10:
                estimated_mag = np.dot(sensitivity_pattern, residual_obs) / sens_norm
                estimated_mag = max(0, estimated_mag)
                predicted_residual = sensitivity_pattern * estimated_mag
                l2_error = np.linalg.norm(residual_obs - predicted_residual)
            else:
                l2_error = np.inf
            
            # Combine correlation and L2 error
            # Higher correlation and lower error = higher score
            scores[node] = correlation - 0.1 * l2_error
        
        return scores
    
    def predict(
        self,
        observed_pressures: np.ndarray,
        base_demand: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Tuple[List[int], np.ndarray, float]:
        """
        Predict top-K leak candidates and estimate magnitude.
        
        Args:
            observed_pressures: Observed sensor pressures.
            base_demand: Base demand vector.
            top_k: Number of top candidates to return.
        
        Returns:
            Tuple of (top_k_nodes, probabilities, estimated_magnitude)
        """
        if base_demand is not None:
            self.compute_sensitivity(base_demand)
        
        scores = self.compute_scores(observed_pressures)
        
        # Convert to probabilities via softmax
        scores_valid = scores.copy()
        scores_valid[scores_valid == -np.inf] = -1e10
        exp_scores = np.exp(scores_valid - np.max(scores_valid))
        probabilities = exp_scores / (exp_scores.sum() + 1e-10)
        
        # Get top-K
        top_k_nodes = np.argsort(scores)[-top_k:][::-1].tolist()
        
        # Estimate magnitude at top node
        top_node = top_k_nodes[0]
        residual_obs = observed_pressures - self.baseline_pressures
        sensitivity_pattern = self.sensitivity[:, top_node]
        sens_norm = np.dot(sensitivity_pattern, sensitivity_pattern)
        if sens_norm > 1e-10:
            estimated_mag = max(0, np.dot(sensitivity_pattern, residual_obs) / sens_norm)
        else:
            estimated_mag = 0.0
        
        return top_k_nodes, probabilities, estimated_mag
    
    def detect_leak(
        self,
        observed_pressures: np.ndarray,
        base_demand: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Detect whether a leak is present.
        
        Uses the magnitude of the residual as a leak indicator.
        
        Args:
            observed_pressures: Observed sensor pressures.
            base_demand: Base demand vector.
            threshold: Threshold for residual magnitude.
        
        Returns:
            Tuple of (has_leak, confidence)
        """
        if base_demand is not None:
            self.compute_sensitivity(base_demand)
        
        residual_obs = observed_pressures - self.baseline_pressures
        residual_magnitude = np.linalg.norm(residual_obs)
        
        # Heuristic: leak if residual magnitude exceeds threshold
        has_leak = residual_magnitude > threshold
        confidence = min(1.0, residual_magnitude / (threshold * 2))
        
        return has_leak, confidence
