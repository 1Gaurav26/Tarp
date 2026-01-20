"""
Explainability Module with Attention Visualization and SHAP Values.

PATENTABLE INNOVATION: Interpretable leak detection with attention maps,
SHAP values, and causal path analysis for explainable AI in water networks.

Key Features:
- Graph attention visualization for leak reasoning
- SHAP-based feature importance for sensor readings
- Causal path analysis showing leak propagation
- Human-readable explanations with natural language generation
"""

from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class AttentionVisualizer:
    """
    Visualizes attention weights in graph neural networks.
    
    Creates interpretable attention maps showing which nodes/edges
    are most important for leak detection.
    """
    
    def __init__(
        self,
        G: nx.Graph,
    ):
        """
        Initialize attention visualizer.
        
        Args:
            G: NetworkX graph.
        """
        self.G = G
    
    def visualize_node_attention(
        self,
        attention_weights: np.ndarray,  # [num_nodes] or [seq_len, num_nodes]
        node_probs: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        title: str = "Node Attention Weights",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize node attention weights.
        
        Args:
            attention_weights: Attention weights for nodes.
            node_probs: Optional leak probabilities for comparison.
            positions: Optional node positions for layout.
            title: Plot title.
            save_path: Optional path to save figure.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Handle temporal attention
        if attention_weights.ndim == 2:
            # Temporal: use mean over time
            attention_weights = attention_weights.mean(axis=0)
        
        # Normalize attention weights
        if attention_weights.max() > 0:
            attention_weights = attention_weights / attention_weights.max()
        
        # Get node positions if not provided
        if positions is None:
            positions = nx.spring_layout(self.G, seed=42)
            positions = np.array([positions[i] for i in range(self.G.number_of_nodes())])
        
        # Plot edges
        for edge in self.G.edges():
            i, j = edge
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                'gray', alpha=0.3, linewidth=0.5,
            )
        
        # Plot nodes with attention weights
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=attention_weights,
            s=200,
            cmap='YlOrRd',
            alpha=0.8,
            edgecolors='black',
            linewidths=1,
        )
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_edge_attention(
        self,
        edge_attention: np.ndarray,  # [num_edges]
        edge_index: np.ndarray,  # [2, num_edges]
        positions: Optional[np.ndarray] = None,
        title: str = "Edge Attention Weights",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize edge attention weights.
        
        Args:
            edge_attention: Attention weights for edges.
            edge_index: Edge indices.
            positions: Optional node positions.
            title: Plot title.
            save_path: Optional path to save figure.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize attention
        if edge_attention.max() > 0:
            edge_attention = edge_attention / edge_attention.max()
        
        # Get positions
        if positions is None:
            pos_dict = nx.spring_layout(self.G, seed=42)
            positions = np.array([pos_dict[i] for i in range(self.G.number_of_nodes())])
        
        # Plot edges with attention-based width and color
        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[:, edge_idx]
            attn = edge_attention[edge_idx]
            
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=plt.cm.YlOrRd(attn),
                linewidth=2 + 3 * attn,
                alpha=0.6,
            )
        
        # Plot nodes
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=100,
            c='gray',
            alpha=0.7,
            edgecolors='black',
        )
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class SHAPExplainer:
    """
    SHAP-based explainer for leak detection predictions.
    
    Computes SHAP values to understand feature importance for
    each prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: torch.Tensor,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained leak detection model.
            background_data: Background dataset for SHAP (training samples).
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.model.eval()
    
    def compute_shap_values(
        self,
        input_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_samples: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SHAP values for input features.
        
        Args:
            input_features: Input node features [batch, num_nodes, feat_dim]
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            num_samples: Number of SHAP samples.
        
        Returns:
            Tuple of (shap_values, base_values)
        """
        # Simplified SHAP computation
        # For full SHAP, would use shap library
        
        batch_size, num_nodes, feat_dim = input_features.shape
        
        # Mean of background data as baseline
        baseline = self.background_data.mean(dim=0, keepdim=True)
        
        # Compute SHAP values (simplified: permutation-based)
        shap_values = torch.zeros_like(input_features)
        
        with torch.no_grad():
            # Get baseline prediction
            baseline_pred, _ = self.model(baseline, edge_index, edge_attr)
            
            # Compute SHAP for each feature
            for b in range(batch_size):
                for n in range(num_nodes):
                    for f in range(feat_dim):
                        # Permutation: replace with baseline
                        input_perm = input_features[b:b+1].clone()
                        input_perm[0, n, f] = baseline[0, n, f]
                        
                        pred_perm, _ = self.model(input_perm, edge_index, edge_attr)
                        
                        # SHAP value = difference in prediction
                        shap_value = (baseline_pred - pred_perm)[0, n]
                        shap_values[b, n, f] = shap_value
        
        return shap_values.numpy(), baseline_pred.numpy()
    
    def visualize_shap_values(
        self,
        shap_values: np.ndarray,  # [batch, num_nodes, feat_dim]
        feature_names: List[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize SHAP values.
        
        Args:
            shap_values: SHAP values.
            feature_names: Names of features.
            save_path: Optional path to save figure.
        
        Returns:
            Matplotlib figure.
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(shap_values.shape[2])]
        
        # Average over batch
        shap_mean = shap_values.mean(axis=0)  # [num_nodes, feat_dim]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bar plot of feature importance
        feat_importance = np.abs(shap_mean).mean(axis=0)  # [feat_dim]
        
        bars = ax.barh(feature_names, feat_importance)
        ax.set_xlabel('Average |SHAP Value|')
        ax.set_title('Feature Importance (SHAP Values)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CausalPathAnalyzer:
    """
    Analyzes causal paths for leak propagation.
    
    Identifies paths from leak location to sensor readings to explain
    how leaks affect sensor measurements.
    """
    
    def __init__(
        self,
        G: nx.Graph,
        simulator: HydraulicSimulator,
    ):
        """
        Initialize causal path analyzer.
        
        Args:
            G: NetworkX graph.
            simulator: Hydraulic simulator.
        """
        self.G = G
        self.simulator = simulator
    
    def analyze_leak_propagation(
        self,
        leak_node: int,
        sensor_nodes: List[int],
        base_demand: np.ndarray,
        leak_magnitude: float = 1.0,
    ) -> Dict[str, any]:
        """
        Analyze how leak propagates to sensors.
        
        Args:
            leak_node: Leak node location.
            sensor_nodes: Sensor node locations.
            base_demand: Base demand vector.
            leak_magnitude: Leak magnitude.
        
        Returns:
            Dictionary with causal paths and pressure changes.
        """
        # Solve without leak
        heads_no_leak = self.simulator.solve(base_demand, leak_node=None)
        
        # Solve with leak
        heads_with_leak = self.simulator.solve(
            base_demand, leak_node=leak_node, leak_magnitude=leak_magnitude
        )
        
        # Pressure changes
        pressure_changes = heads_with_leak - heads_no_leak
        
        # Find paths from leak to sensors
        causal_paths = {}
        path_lengths = {}
        
        for sensor in sensor_nodes:
            try:
                # Shortest path
                path = nx.shortest_path(self.G, leak_node, sensor)
                causal_paths[sensor] = path
                path_lengths[sensor] = len(path) - 1
            except nx.NetworkXNoPath:
                causal_paths[sensor] = None
                path_lengths[sensor] = float('inf')
        
        # Pressure change along paths
        path_pressure_changes = {}
        for sensor, path in causal_paths.items():
            if path:
                path_changes = [pressure_changes[n] for n in path]
                path_pressure_changes[sensor] = path_changes
        
        return {
            "causal_paths": causal_paths,
            "path_lengths": path_lengths,
            "pressure_changes": pressure_changes,
            "path_pressure_changes": path_pressure_changes,
            "sensor_pressure_changes": pressure_changes[sensor_nodes],
        }
    
    def visualize_causal_paths(
        self,
        leak_node: int,
        sensor_nodes: List[int],
        causal_analysis: Dict[str, any],
        positions: Optional[np.ndarray] = None,
        title: str = "Leak Propagation Paths",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize causal paths from leak to sensors.
        
        Args:
            leak_node: Leak node.
            sensor_nodes: Sensor nodes.
            causal_analysis: Output from analyze_leak_propagation.
            positions: Optional node positions.
            title: Plot title.
            save_path: Optional path to save figure.
        
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if positions is None:
            pos_dict = nx.spring_layout(self.G, seed=42)
            positions = np.array([pos_dict[i] for i in range(self.G.number_of_nodes())])
        
        # Plot all edges (gray)
        for edge in self.G.edges():
            i, j = edge
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                'gray', alpha=0.2, linewidth=0.5,
            )
        
        # Plot causal paths (colored)
        causal_paths = causal_analysis["causal_paths"]
        path_pressure_changes = causal_analysis["path_pressure_changes"]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sensor_nodes)))
        
        for idx, sensor in enumerate(sensor_nodes):
            path = causal_paths.get(sensor)
            if path:
                # Plot path
                path_positions = positions[path]
                ax.plot(
                    path_positions[:, 0],
                    path_positions[:, 1],
                    color=colors[idx],
                    linewidth=3,
                    alpha=0.7,
                    label=f"Path to Sensor {sensor}",
                )
        
        # Plot leak node (red star)
        ax.scatter(
            positions[leak_node, 0],
            positions[leak_node, 1],
            s=500,
            marker='*',
            c='red',
            edgecolors='black',
            linewidths=2,
            label='Leak',
            zorder=10,
        )
        
        # Plot sensor nodes (blue circles)
        for sensor in sensor_nodes:
            ax.scatter(
                positions[sensor, 0],
                positions[sensor, 1],
                s=300,
                c='blue',
                edgecolors='black',
                linewidths=2,
                marker='o',
                zorder=10,
            )
        
        ax.legend()
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ExplanationGenerator:
    """
    Generates human-readable explanations for leak predictions.
    
    Combines attention weights, SHAP values, and causal paths into
    natural language explanations.
    """
    
    def __init__(
        self,
        G: nx.Graph,
        simulator: Optional[HydraulicSimulator] = None,
    ):
        """
        Initialize explanation generator.
        
        Args:
            G: NetworkX graph.
            simulator: Hydraulic simulator (optional).
        """
        self.G = G
        self.simulator = simulator
    
    def generate_explanation(
        self,
        predicted_leak_node: int,
        leak_probability: float,
        top_k_nodes: List[int],
        attention_weights: Optional[np.ndarray] = None,
        shap_values: Optional[np.ndarray] = None,
        causal_analysis: Optional[Dict[str, any]] = None,
    ) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            predicted_leak_node: Predicted leak node.
            leak_probability: Leak probability.
            top_k_nodes: Top-K candidate nodes.
            attention_weights: Optional attention weights.
            shap_values: Optional SHAP values.
            causal_analysis: Optional causal analysis.
        
        Returns:
            Natural language explanation string.
        """
        explanation_parts = []
        
        # Main prediction
        explanation_parts.append(
            f"The model predicts a leak at Node {predicted_leak_node} "
            f"with probability {leak_probability:.2%}."
        )
        
        # Confidence level
        if leak_probability > 0.8:
            confidence = "high"
        elif leak_probability > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        explanation_parts.append(
            f"This is a {confidence} confidence prediction."
        )
        
        # Top-K candidates
        if len(top_k_nodes) > 1:
            explanation_parts.append(
                f"Other candidate locations include: {', '.join([f'Node {n}' for n in top_k_nodes[1:min(4, len(top_k_nodes))]])}."
            )
        
        # Attention-based reasoning
        if attention_weights is not None:
            if attention_weights.ndim == 2:
                attention_weights = attention_weights.mean(axis=0)
            
            # Find nodes with high attention
            top_attention_nodes = np.argsort(attention_weights)[-3:][::-1]
            explanation_parts.append(
                f"The model focused attention on nodes {', '.join([f'{n}' for n in top_attention_nodes])} "
                f"when making this prediction."
            )
        
        # SHAP-based feature importance
        if shap_values is not None:
            # Average SHAP values
            shap_mean = np.abs(shap_values).mean(axis=0)
            if shap_mean.ndim > 1:
                shap_mean = shap_mean.mean(axis=0)
            
            max_feature_idx = np.argmax(shap_mean)
            explanation_parts.append(
                f"The most important feature for this prediction is feature {max_feature_idx}."
            )
        
        # Causal reasoning
        if causal_analysis and self.simulator is not None:
            causal_paths = causal_analysis.get("causal_paths", {})
            sensor_nodes = list(causal_paths.keys())
            
            if sensor_nodes:
                shortest_path_length = min([
                    causal_analysis["path_lengths"][s]
                    for s in sensor_nodes if causal_analysis["path_lengths"][s] < float('inf')
                ])
                
                explanation_parts.append(
                    f"The leak propagates through the network, with the shortest path "
                    f"to sensors being {shortest_path_length} hops."
                )
        
        # Combine explanation
        explanation = " ".join(explanation_parts)
        
        return explanation

