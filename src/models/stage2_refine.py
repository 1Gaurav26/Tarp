"""
Stage 2 Refinement for Leak Magnitude Estimation.

Two approaches:
1. PhysicsBasedRefiner: Uses closed-form linear algebra solution
2. Stage2Refiner: Learned MLP regressor on subgraph features
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from ..sim.simulator import HydraulicSimulator


class PhysicsBasedRefiner:
    """
    Physics-based leak magnitude estimation.
    
    Uses the linear relationship between leak magnitude and sensor pressures
    to estimate leak magnitude via least squares.
    """
    
    def __init__(
        self,
        simulator: HydraulicSimulator,
        sensor_nodes: List[int],
    ):
        """
        Initialize the physics-based refiner.
        
        Args:
            simulator: Hydraulic simulator instance.
            sensor_nodes: List of sensor node indices.
        """
        self.simulator = simulator
        self.sensor_nodes = sensor_nodes
    
    def estimate_magnitude(
        self,
        candidate_node: int,
        observed_pressures: np.ndarray,
        base_demand: np.ndarray,
    ) -> float:
        """
        Estimate leak magnitude at a candidate node.
        
        Args:
            candidate_node: Node where leak is hypothesized.
            observed_pressures: Observed pressures at sensor nodes.
            base_demand: Base demand vector.
        
        Returns:
            Estimated leak magnitude.
        """
        return self.simulator.estimate_leak_magnitude_physics(
            candidate_node=candidate_node,
            sensor_nodes=self.sensor_nodes,
            observed_pressures=observed_pressures,
            base_demand=base_demand,
        )
    
    def refine_top_k(
        self,
        top_k_nodes: List[int],
        observed_pressures: np.ndarray,
        base_demand: np.ndarray,
    ) -> Tuple[int, float, Dict[int, float]]:
        """
        Estimate magnitudes for top-K candidates and select the best one.
        
        The best candidate is selected based on which one produces
        the smallest residual error at sensors when the estimated
        magnitude is applied.
        
        Args:
            top_k_nodes: List of top-K candidate nodes.
            observed_pressures: Observed pressures at sensor nodes.
            base_demand: Base demand vector.
        
        Returns:
            Tuple of (best_node, best_magnitude, all_estimates)
        """
        estimates = {}
        errors = {}
        
        for node in top_k_nodes:
            # Estimate magnitude
            mag = self.estimate_magnitude(node, observed_pressures, base_demand)
            estimates[node] = mag
            
            # Compute residual error with this estimate
            h_pred = self.simulator.solve(base_demand, leak_node=node, leak_magnitude=mag)
            p_pred = h_pred[self.sensor_nodes]
            error = np.mean((p_pred - observed_pressures) ** 2)
            errors[node] = error
        
        # Select node with minimum error
        best_node = min(errors, key=errors.get)
        best_magnitude = estimates[best_node]
        
        return best_node, best_magnitude, estimates


class Stage2Refiner(nn.Module):
    """
    Learned Stage 2 refiner using MLP on local features.
    
    Takes features from the candidate node and its local neighborhood
    and predicts the leak magnitude.
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 32,
        num_neighbors: int = 5,
    ):
        """
        Initialize the Stage 2 refiner.
        
        Args:
            input_dim: Dimension of node embeddings from Stage 1.
            hidden_dim: Hidden dimension.
            num_neighbors: Number of neighbor features to include.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        
        # Feature dimension: candidate + aggregated neighbors + global stats
        feat_dim = input_dim * 3  # candidate, neighbor_mean, neighbor_max
        
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive output
        )
    
    def forward(
        self,
        candidate_embedding: torch.Tensor,
        neighbor_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            candidate_embedding: Embedding of candidate node [batch, input_dim]
            neighbor_embeddings: Embeddings of neighbors [batch, num_neighbors, input_dim]
        
        Returns:
            Predicted leak magnitude [batch, 1]
        """
        # Aggregate neighbors
        neighbor_mean = neighbor_embeddings.mean(dim=1)
        neighbor_max = neighbor_embeddings.max(dim=1)[0]
        
        # Concatenate features
        features = torch.cat([
            candidate_embedding,
            neighbor_mean,
            neighbor_max,
        ], dim=-1)
        
        return self.mlp(features)
    
    def extract_features(
        self,
        G: nx.Graph,
        node_embeddings: torch.Tensor,
        candidate_node: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features for a candidate node.
        
        Args:
            G: NetworkX graph.
            node_embeddings: All node embeddings [num_nodes, input_dim]
            candidate_node: Candidate node index.
        
        Returns:
            Tuple of (candidate_embedding, neighbor_embeddings)
        """
        candidate_emb = node_embeddings[candidate_node].unsqueeze(0)
        
        # Get neighbors
        neighbors = list(G.neighbors(candidate_node))
        
        if len(neighbors) == 0:
            # No neighbors, use zero embeddings
            neighbor_embs = torch.zeros(1, self.num_neighbors, self.input_dim)
        else:
            # Pad or truncate to num_neighbors
            if len(neighbors) < self.num_neighbors:
                neighbors = neighbors + [neighbors[0]] * (self.num_neighbors - len(neighbors))
            else:
                neighbors = neighbors[:self.num_neighbors]
            
            neighbor_embs = node_embeddings[neighbors].unsqueeze(0)
        
        return candidate_emb, neighbor_embs


class HybridRefiner:
    """
    Hybrid refiner that combines physics-based and learned approaches.
    
    Uses physics-based estimation as the primary method, with optional
    learned correction.
    """
    
    def __init__(
        self,
        simulator: HydraulicSimulator,
        sensor_nodes: List[int],
        learned_refiner: Optional[Stage2Refiner] = None,
        use_physics: bool = True,
    ):
        """
        Initialize hybrid refiner.
        
        Args:
            simulator: Hydraulic simulator.
            sensor_nodes: Sensor node indices.
            learned_refiner: Optional learned refiner for correction.
            use_physics: Whether to use physics-based estimation.
        """
        self.physics_refiner = PhysicsBasedRefiner(simulator, sensor_nodes)
        self.learned_refiner = learned_refiner
        self.use_physics = use_physics
    
    def refine(
        self,
        top_k_nodes: List[int],
        observed_pressures: np.ndarray,
        base_demand: np.ndarray,
        node_embeddings: Optional[torch.Tensor] = None,
        G: Optional[nx.Graph] = None,
    ) -> Tuple[int, float, Dict[int, float]]:
        """
        Refine predictions for top-K candidates.
        
        Args:
            top_k_nodes: Top-K candidate nodes from Stage 1.
            observed_pressures: Observed sensor pressures.
            base_demand: Base demand vector.
            node_embeddings: Optional embeddings for learned approach.
            G: NetworkX graph (required for learned approach).
        
        Returns:
            Tuple of (best_node, best_magnitude, all_estimates)
        """
        if self.use_physics:
            return self.physics_refiner.refine_top_k(
                top_k_nodes, observed_pressures, base_demand
            )
        elif self.learned_refiner is not None and node_embeddings is not None and G is not None:
            # Use learned refiner
            estimates = {}
            
            self.learned_refiner.eval()
            with torch.no_grad():
                for node in top_k_nodes:
                    cand_emb, neighbor_embs = self.learned_refiner.extract_features(
                        G, node_embeddings, node
                    )
                    mag = self.learned_refiner(cand_emb, neighbor_embs).item()
                    estimates[node] = mag
            
            # Select node with highest Stage 1 probability (already ranked)
            best_node = top_k_nodes[0]
            best_magnitude = estimates[best_node]
            
            return best_node, best_magnitude, estimates
        else:
            raise ValueError("No valid refinement method available")


def train_stage2_on_correct_predictions(
    refiner: Stage2Refiner,
    dataset,
    gnn_model,
    G: nx.Graph,
    epochs: int = 20,
    lr: float = 0.001,
) -> List[float]:
    """
    Train Stage 2 refiner on samples where Stage 1 correctly identifies the leak node.
    
    Args:
        refiner: Stage 2 refiner model.
        dataset: Training dataset.
        gnn_model: Trained Stage 1 GNN.
        G: NetworkX graph.
        epochs: Number of training epochs.
        lr: Learning rate.
    
    Returns:
        List of training losses per epoch.
    """
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    gnn_model.eval()
    losses = []
    
    for epoch in range(epochs):
        refiner.train()
        epoch_loss = 0.0
        num_samples = 0
        
        for i in range(len(dataset)):
            sample = dataset[i]
            
            # Skip no-leak samples
            if sample["has_leak"] < 0.5:
                continue
            
            true_node = sample["leak_node"].item()
            true_mag = sample["leak_magnitude"].item()
            
            # Get Stage 1 predictions
            with torch.no_grad():
                node_probs, _ = gnn_model.predict(
                    sample["node_features"].unsqueeze(0),
                    sample["edge_index"],
                    sample["edge_attr"],
                )
                pred_node = node_probs.argmax().item()
            
            # Only train on correctly localized samples
            if pred_node != true_node:
                continue
            
            # Get embeddings (we'd need to store intermediate representations)
            # For simplicity, use node features as proxy
            node_embs = sample["node_features"]
            
            cand_emb, neighbor_embs = refiner.extract_features(G, node_embs, true_node)
            
            optimizer.zero_grad()
            pred_mag = refiner(cand_emb, neighbor_embs)
            loss = criterion(pred_mag, torch.tensor([[true_mag]]))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_samples += 1
        
        if num_samples > 0:
            avg_loss = epoch_loss / num_samples
            losses.append(avg_loss)
    
    return losses
