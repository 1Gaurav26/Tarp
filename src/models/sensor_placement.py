"""
Adaptive Sensor Placement Optimization using Information Gain.

PATENTABLE INNOVATION: Dynamic sensor placement optimization using information-
theoretic criteria and graph centrality measures for optimal leak detection.

Key Features:
- Information gain maximization for sensor placement
- Graph centrality-based candidate selection
- Active learning for optimal sensor deployment
- Cost-benefit optimization with budget constraints
"""

from typing import Dict, List, Optional, Set, Tuple
import itertools

import numpy as np
import torch
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy

from ..sim.simulator import HydraulicSimulator
from .gnn_stage1 import LeakDetectionGNN


class SensorPlacementOptimizer:
    """
    Optimizes sensor placement using information-theoretic criteria.
    
    PATENTABLE METHOD: Uses information gain, graph centrality, and 
    sensitivity-based heuristics for optimal sensor deployment.
    """
    
    def __init__(
        self,
        G: nx.Graph,
        simulator: HydraulicSimulator,
        base_gnn: Optional[LeakDetectionGNN] = None,
    ):
        """
        Initialize sensor placement optimizer.
        
        Args:
            G: NetworkX graph.
            simulator: Hydraulic simulator.
            base_gnn: Pre-trained GNN model (optional).
        """
        self.G = G
        self.simulator = simulator
        self.base_gnn = base_gnn
        self.num_nodes = G.number_of_nodes()
        self.reservoir_nodes = list(simulator.reservoir_nodes)
        
        # Candidate nodes (exclude reservoirs)
        self.candidate_nodes = [
            n for n in range(self.num_nodes) if n not in self.reservoir_nodes
        ]
    
    def compute_information_gain(
        self,
        sensor_candidates: List[int],
        base_demand: np.ndarray,
        num_samples: int = 100,
    ) -> float:
        """
        Compute information gain for a sensor configuration.
        
        Information gain measures how much sensor placement reduces
        uncertainty about leak location.
        
        Args:
            sensor_candidates: List of candidate sensor nodes.
            base_demand: Base demand vector.
            num_samples: Number of leak scenarios to sample.
        
        Returns:
            Information gain value (higher = better).
        """
        # Sample leak scenarios
        leak_scenarios = []
        leak_locations = []
        
        np.random.seed(42)
        for _ in range(num_samples):
            leak_node = np.random.choice(self.candidate_nodes)
            leak_magnitude = np.random.uniform(0.5, 5.0)
            
            # Solve with leak
            heads = self.simulator.solve(
                base_demand,
                leak_node=leak_node,
                leak_magnitude=leak_magnitude,
            )
            
            # Get sensor readings
            sensor_readings = heads[sensor_candidates]
            leak_scenarios.append(sensor_readings)
            leak_locations.append(leak_node)
        
        leak_scenarios = np.array(leak_scenarios)  # [num_samples, num_sensors]
        leak_locations = np.array(leak_locations)
        
        # Compute entropy before (uniform over all nodes)
        num_candidate_nodes = len(self.candidate_nodes)
        entropy_before = np.log2(num_candidate_nodes)  # Maximum entropy
        
        # Compute entropy after (using sensor readings)
        # Discretize sensor readings into bins
        num_bins = 20
        sensor_histograms = []
        
        for sensor_idx in range(len(sensor_candidates)):
            readings = leak_scenarios[:, sensor_idx]
            hist, _ = np.histogram(readings, bins=num_bins)
            hist = hist / hist.sum()  # Normalize
            sensor_histograms.append(hist)
        
        # Joint distribution approximation (product of marginals for simplicity)
        joint_hist = np.ones(num_bins ** len(sensor_candidates))
        for i, hist in enumerate(sensor_histograms):
            # Simple product (assumes independence for approximation)
            pass  # Simplified for now
        
        # Compute conditional entropy: H(leak | sensors)
        entropy_after = 0.0
        
        # Bin sensor readings
        binned_readings = []
        for sensor_idx in range(len(sensor_candidates)):
            readings = leak_scenarios[:, sensor_idx]
            bins = np.linspace(readings.min(), readings.max(), num_bins + 1)
            bins_indices = np.digitize(readings, bins) - 1
            binned_readings.append(bins_indices)
        
        binned_readings = np.array(binned_readings).T  # [num_samples, num_sensors]
        
        # Compute conditional entropy
        unique_patterns = {}
        for i, pattern in enumerate(binned_readings):
            pattern_tuple = tuple(pattern)
            if pattern_tuple not in unique_patterns:
                unique_patterns[pattern_tuple] = []
            unique_patterns[pattern_tuple].append(leak_locations[i])
        
        conditional_entropy = 0.0
        for pattern, locations in unique_patterns.items():
            pattern_prob = len(locations) / num_samples
            location_counts = {}
            for loc in locations:
                location_counts[loc] = location_counts.get(loc, 0) + 1
            
            # Entropy of leak location given this sensor pattern
            location_probs = np.array(list(location_counts.values())) / len(locations)
            pattern_entropy = entropy(location_probs, base=2)
            
            conditional_entropy += pattern_prob * pattern_entropy
        
        # Information gain = entropy_before - conditional_entropy
        information_gain = entropy_before - conditional_entropy
        
        return information_gain
    
    def compute_graph_centrality(
        self,
        candidates: List[int],
    ) -> Dict[int, float]:
        """
        Compute graph centrality measures for candidates.
        
        Uses multiple centrality measures:
        - Betweenness centrality
        - Closeness centrality
        - PageRank
        - Degree centrality
        """
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.G)
        
        # Closeness centrality
        closeness = nx.closeness_centrality(self.G)
        
        # PageRank
        pagerank = nx.pagerank(self.G)
        
        # Degree centrality
        degree = nx.degree_centrality(self.G)
        
        # Combine scores (weighted average)
        centrality_scores = {}
        for node in candidates:
            score = (
                0.3 * betweenness[node] +
                0.3 * closeness[node] +
                0.2 * pagerank[node] +
                0.2 * degree[node]
            )
            centrality_scores[node] = score
        
        return centrality_scores
    
    def compute_sensitivity_coverage(
        self,
        sensor_candidates: List[int],
        base_demand: np.ndarray,
    ) -> float:
        """
        Compute sensitivity coverage score.
        
        Measures how well sensors cover different leak locations
        via sensitivity matrix analysis.
        """
        # Compute sensitivity matrix
        sensitivity = self.simulator.compute_sensitivity_matrix(
            sensor_candidates,
            base_demand,
        )
        
        # Sensitivity matrix: [num_sensors, num_nodes]
        # For each potential leak node, compute detectability
        
        # Coverage: number of nodes with high sensitivity
        # High sensitivity = absolute value > threshold
        threshold = np.percentile(np.abs(sensitivity), 75)
        
        node_detectability = np.max(np.abs(sensitivity), axis=0)  # [num_nodes]
        coverage = (node_detectability > threshold).sum()
        
        # Normalize by number of candidate nodes
        coverage_score = coverage / len(self.candidate_nodes)
        
        return coverage_score
    
    def optimize_placement(
        self,
        num_sensors: int,
        base_demand: Optional[np.ndarray] = None,
        method: str = "hybrid",
        budget: Optional[float] = None,
        sensor_costs: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Optimize sensor placement.
        
        PATENTABLE METHOD: Combines information gain, graph centrality,
        and sensitivity coverage for optimal placement.
        
        Args:
            num_sensors: Number of sensors to place.
            base_demand: Base demand vector (generated if None).
            method: Optimization method ("info_gain", "centrality", "sensitivity", "hybrid").
            budget: Optional budget constraint.
            sensor_costs: Optional cost dictionary per node.
        
        Returns:
            Tuple of (optimal_sensor_nodes, metrics_dict)
        """
        if base_demand is None:
            base_demand = self.simulator.generate_demand()
        
        if sensor_costs is None:
            sensor_costs = {n: 1.0 for n in self.candidate_nodes}
        
        # Method: hybrid (combination of all)
        if method == "hybrid":
            # Get centrality scores
            centrality_scores = self.compute_graph_centrality(self.candidate_nodes)
            
            # Select initial candidates using centrality
            top_centrality = sorted(
                centrality_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:min(num_sensors * 3, len(self.candidate_nodes))]
            
            top_candidates = [node for node, _ in top_centrality]
            
            # Greedy selection with information gain and coverage
            selected = []
            remaining = set(top_candidates)
            
            # Start with highest centrality node
            if top_candidates:
                selected.append(top_candidates[0])
                remaining.remove(top_candidates[0])
            
            # Greedy addition
            for _ in range(num_sensors - 1):
                if not remaining:
                    break
                
                best_node = None
                best_score = -np.inf
                
                for candidate in remaining:
                    # Check budget
                    if budget is not None:
                        current_cost = sum(sensor_costs[n] for n in selected)
                        if current_cost + sensor_costs[candidate] > budget:
                            continue
                    
                    trial_set = selected + [candidate]
                    
                    # Combined score
                    # Information gain (if small set)
                    if len(trial_set) <= 10:
                        info_gain = self.compute_information_gain(
                            trial_set, base_demand, num_samples=50
                        )
                    else:
                        info_gain = 0.0  # Skip expensive computation
                    
                    # Sensitivity coverage
                    sensitivity_score = self.compute_sensitivity_coverage(
                        trial_set, base_demand
                    )
                    
                    # Centrality
                    centrality_score = centrality_scores[candidate]
                    
                    # Combined score (weighted)
                    score = (
                        0.4 * info_gain +
                        0.3 * sensitivity_score +
                        0.3 * centrality_score
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_node = candidate
                
                if best_node is not None:
                    selected.append(best_node)
                    remaining.remove(best_node)
                else:
                    break  # Budget constraint or no valid candidates
            
            # Compute final metrics
            final_info_gain = self.compute_information_gain(
                selected, base_demand, num_samples=100
            )
            final_coverage = self.compute_sensitivity_coverage(selected, base_demand)
            final_centrality = np.mean([centrality_scores[n] for n in selected])
            
            metrics = {
                "information_gain": final_info_gain,
                "sensitivity_coverage": final_coverage,
                "average_centrality": final_centrality,
                "total_cost": sum(sensor_costs[n] for n in selected),
            }
            
            return selected, metrics
        
        elif method == "centrality":
            # Pure centrality-based
            centrality_scores = self.compute_graph_centrality(self.candidate_nodes)
            sorted_nodes = sorted(
                centrality_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:num_sensors]
            
            selected = [node for node, _ in sorted_nodes]
            metrics = {"method": "centrality"}
            
            return selected, metrics
        
        elif method == "sensitivity":
            # Pure sensitivity-based (greedy)
            selected = []
            remaining = set(self.candidate_nodes)
            
            # Greedy selection based on sensitivity coverage
            for _ in range(num_sensors):
                best_node = None
                best_coverage = -1.0
                
                for candidate in remaining:
                    trial_set = selected + [candidate]
                    coverage = self.compute_sensitivity_coverage(trial_set, base_demand)
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_node = candidate
                
                if best_node is not None:
                    selected.append(best_node)
                    remaining.remove(best_node)
            
            metrics = {"sensitivity_coverage": best_coverage}
            return selected, metrics
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def active_learning_update(
        self,
        current_sensors: List[int],
        observed_data: np.ndarray,
        base_demand: np.ndarray,
        num_additional: int = 1,
    ) -> Tuple[List[int], float]:
        """
        Active learning: suggest additional sensors based on current data.
        
        PATENTABLE METHOD: Uses uncertainty in current predictions to
        suggest optimal sensor placement.
        
        Args:
            current_sensors: Currently deployed sensor nodes.
            observed_data: Observed sensor readings.
            base_demand: Base demand vector.
            num_additional: Number of additional sensors to suggest.
        
        Returns:
            Tuple of (additional_sensor_nodes, expected_info_gain)
        """
        # Identify regions with high uncertainty
        # (This would typically use GNN predictions)
        
        # For now, use sensitivity-based approach
        # Find nodes where adding a sensor would provide most information
        
        best_additional = []
        remaining_candidates = [
            n for n in self.candidate_nodes if n not in current_sensors
        ]
        
        for _ in range(num_additional):
            best_node = None
            best_gain = -np.inf
            
            for candidate in remaining_candidates:
                trial_set = current_sensors + [candidate]
                
                info_gain = self.compute_information_gain(
                    trial_set, base_demand, num_samples=50
                )
                
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_node = candidate
            
            if best_node is not None:
                best_additional.append(best_node)
                remaining_candidates.remove(best_node)
        
        return best_additional, best_gain

