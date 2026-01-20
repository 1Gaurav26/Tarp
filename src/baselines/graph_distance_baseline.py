"""
Graph Distance Baseline for Water Leak Localization.

This baseline uses graph distance from anomalous sensors to rank
potential leak locations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


class GraphDistanceBaseline:
    """
    Baseline that localizes leaks based on graph distance from anomalous sensors.
    
    Logic:
    1. Identify sensors with anomalous pressure readings
    2. For each node, compute weighted sum of distances to anomalous sensors
    3. Nodes closer to multiple anomalous sensors are more likely leak locations
    """
    
    def __init__(
        self,
        G: nx.Graph,
        sensor_nodes: List[int],
        reservoir_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the baseline.
        
        Args:
            G: NetworkX graph.
            sensor_nodes: List of sensor node indices.
            reservoir_nodes: Optional list of reservoir nodes to exclude.
        """
        self.G = G
        self.sensor_nodes = sensor_nodes
        self.reservoir_nodes = set(reservoir_nodes) if reservoir_nodes else set()
        self.num_nodes = G.number_of_nodes()
        
        # Precompute shortest path distances
        self.distances = dict(nx.all_pairs_shortest_path_length(G))
    
    def identify_anomalous_sensors(
        self,
        observed_pressures: np.ndarray,
        baseline_pressures: np.ndarray,
        threshold_std: float = 2.0,
    ) -> List[int]:
        """
        Identify sensors with anomalous readings.
        
        Args:
            observed_pressures: Observed sensor pressures.
            baseline_pressures: Expected baseline pressures.
            threshold_std: Number of std deviations for anomaly detection.
        
        Returns:
            List of anomalous sensor indices (in sensor_nodes list).
        """
        residuals = observed_pressures - baseline_pressures
        residual_std = np.std(residuals)
        
        if residual_std < 1e-8:
            return []
        
        anomalous_indices = np.where(np.abs(residuals) > threshold_std * residual_std)[0]
        return [self.sensor_nodes[i] for i in anomalous_indices]
    
    def compute_scores(
        self,
        observed_pressures: np.ndarray,
        baseline_pressures: np.ndarray,
        threshold_std: float = 2.0,
    ) -> np.ndarray:
        """
        Compute leak probability scores based on graph distance.
        
        Args:
            observed_pressures: Observed sensor pressures.
            baseline_pressures: Expected baseline pressures.
            threshold_std: Threshold for anomaly detection.
        
        Returns:
            Array of scores for each node.
        """
        anomalous_sensors = self.identify_anomalous_sensors(
            observed_pressures, baseline_pressures, threshold_std
        )
        
        if len(anomalous_sensors) == 0:
            # No anomalies detected, return uniform scores
            return np.ones(self.num_nodes) / self.num_nodes
        
        # Compute residuals for weighting
        residuals = observed_pressures - baseline_pressures
        sensor_to_idx = {s: i for i, s in enumerate(self.sensor_nodes)}
        
        scores = np.zeros(self.num_nodes)
        
        for node in range(self.num_nodes):
            if node in self.reservoir_nodes:
                scores[node] = -np.inf
                continue
            
            weighted_score = 0.0
            for sensor in anomalous_sensors:
                dist = self.distances[node].get(sensor, self.num_nodes)
                # Weight by inverse distance and residual magnitude
                weight = np.abs(residuals[sensor_to_idx[sensor]])
                if dist > 0:
                    weighted_score += weight / dist
                else:
                    weighted_score += weight * 10  # Same node, high score
            
            scores[node] = weighted_score
        
        return scores
    
    def predict(
        self,
        observed_pressures: np.ndarray,
        baseline_pressures: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Predict top-K leak candidates.
        
        Args:
            observed_pressures: Observed sensor pressures.
            baseline_pressures: Expected baseline pressures.
            top_k: Number of top candidates.
        
        Returns:
            Tuple of (top_k_nodes, probabilities)
        """
        scores = self.compute_scores(observed_pressures, baseline_pressures)
        
        # Convert to probabilities
        scores_valid = scores.copy()
        scores_valid[scores_valid == -np.inf] = -1e10
        
        # Softmax
        exp_scores = np.exp(scores_valid - np.max(scores_valid))
        probabilities = exp_scores / (exp_scores.sum() + 1e-10)
        
        # Get top-K
        top_k_nodes = np.argsort(scores)[-top_k:][::-1].tolist()
        
        return top_k_nodes, probabilities


class CorrelationBaseline:
    """
    Alternative baseline using correlation between sensor readings.
    
    Analyzes correlation patterns in sensor readings to identify
    anomalous regions that might indicate a leak.
    """
    
    def __init__(
        self,
        G: nx.Graph,
        sensor_nodes: List[int],
    ):
        """
        Initialize the baseline.
        
        Args:
            G: NetworkX graph.
            sensor_nodes: List of sensor node indices.
        """
        self.G = G
        self.sensor_nodes = sensor_nodes
        self.num_nodes = G.number_of_nodes()
        
        # Compute sensor-to-node distances
        self.sensor_distances = {}
        for node in range(self.num_nodes):
            self.sensor_distances[node] = []
            for sensor in sensor_nodes:
                try:
                    dist = nx.shortest_path_length(G, node, sensor)
                except nx.NetworkXNoPath:
                    dist = self.num_nodes
                self.sensor_distances[node].append(dist)
            self.sensor_distances[node] = np.array(self.sensor_distances[node])
    
    def compute_scores(
        self,
        observed_pressures: np.ndarray,
        baseline_pressures: np.ndarray,
    ) -> np.ndarray:
        """
        Compute scores based on distance-weighted residuals.
        
        For each candidate node, computes a score based on how well
        the residual pattern matches what we'd expect from a leak there.
        
        Args:
            observed_pressures: Observed sensor pressures.
            baseline_pressures: Expected baseline pressures.
        
        Returns:
            Array of scores for each node.
        """
        residuals = observed_pressures - baseline_pressures
        
        scores = np.zeros(self.num_nodes)
        
        for node in range(self.num_nodes):
            distances = self.sensor_distances[node]
            
            # Weight residuals by inverse distance
            # Sensors closer to the leak should show larger residuals
            weights = 1.0 / (distances + 1)
            
            # Score: correlation between weights and absolute residuals
            if np.std(weights) > 1e-8 and np.std(np.abs(residuals)) > 1e-8:
                score = np.corrcoef(weights, np.abs(residuals))[0, 1]
                if np.isnan(score):
                    score = 0.0
            else:
                score = 0.0
            
            scores[node] = score
        
        return scores
    
    def predict(
        self,
        observed_pressures: np.ndarray,
        baseline_pressures: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Predict top-K leak candidates.
        
        Args:
            observed_pressures: Observed sensor pressures.
            baseline_pressures: Expected baseline pressures.
            top_k: Number of top candidates.
        
        Returns:
            Tuple of (top_k_nodes, probabilities)
        """
        scores = self.compute_scores(observed_pressures, baseline_pressures)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / (exp_scores.sum() + 1e-10)
        
        # Get top-K
        top_k_nodes = np.argsort(scores)[-top_k:][::-1].tolist()
        
        return top_k_nodes, probabilities
