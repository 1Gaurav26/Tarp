"""
MLP Baseline for Water Leak Localization.

A simple fully-connected neural network that takes sensor readings
and predicts leak location and magnitude.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline for leak localization.
    
    Takes sensor readings as input and predicts:
    - Per-node leak probability
    - Global no-leak probability
    - Leak magnitude (if leak detected)
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_sensors: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """
        Initialize the MLP baseline.
        
        Args:
            num_nodes: Number of nodes in the graph.
            num_sensors: Number of sensor nodes.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_sensors = num_sensors
        
        # Shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(num_sensors, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Node-level prediction head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        
        # Global no-leak prediction head
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Magnitude prediction head
        self.magnitude_head = nn.Sequential(
            nn.Linear(hidden_dim + num_nodes, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive output
        )
    
    def forward(
        self,
        sensor_readings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sensor_readings: Sensor pressure readings [batch, num_sensors]
        
        Returns:
            Tuple of:
            - node_logits: Per-node leak logits [batch, num_nodes]
            - no_leak_logit: No-leak logit [batch, 1]
            - magnitude: Predicted leak magnitude [batch, 1]
        """
        # Extract features
        features = self.feature_net(sensor_readings)
        
        # Node predictions
        node_logits = self.node_head(features)
        
        # Global no-leak prediction
        no_leak_logit = self.global_head(features)
        
        # Magnitude prediction (conditioned on node predictions)
        node_probs = torch.softmax(node_logits, dim=-1)
        magnitude_input = torch.cat([features, node_probs], dim=-1)
        magnitude = self.magnitude_head(magnitude_input)
        
        return node_logits, no_leak_logit, magnitude
    
    def predict(
        self,
        sensor_readings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions (probabilities).
        
        Args:
            sensor_readings: Sensor pressure readings [batch, num_sensors]
        
        Returns:
            Tuple of:
            - node_probs: Per-node leak probabilities [batch, num_nodes]
            - no_leak_prob: No-leak probability [batch, 1]
            - magnitude: Predicted leak magnitude [batch, 1]
        """
        node_logits, no_leak_logit, magnitude = self.forward(sensor_readings)
        node_probs = torch.sigmoid(node_logits)
        no_leak_prob = torch.sigmoid(no_leak_logit)
        return node_probs, no_leak_prob, magnitude


class MLPBaselineTrainer:
    """Trainer for the MLP baseline."""
    
    def __init__(
        self,
        model: MLPBaseline,
        sensor_nodes: List[int],
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: MLP baseline model.
            sensor_nodes: List of sensor node indices.
            learning_rate: Learning rate.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.sensor_nodes = sensor_nodes
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.focal_loss = FocalLossSimple(gamma=2.0)
    
    def extract_sensor_readings(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract sensor readings from node features.
        
        Args:
            node_features: Full node features [batch, num_nodes, feat_dim]
        
        Returns:
            Sensor readings [batch, num_sensors]
        """
        # First feature dimension is the pressure reading
        pressures = node_features[:, :, 0]  # [batch, num_nodes]
        sensor_readings = pressures[:, self.sensor_nodes]  # [batch, num_sensors]
        return sensor_readings
    
    def train_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Perform a training step.
        
        Args:
            batch: Batch of data.
        
        Returns:
            Dictionary of losses.
        """
        self.model.train()
        
        node_features = batch["node_features"].to(self.device)
        y_node = batch["y_node"].to(self.device)
        has_leak = batch["has_leak"].to(self.device)
        leak_magnitude = batch["leak_magnitude"].to(self.device)
        
        sensor_readings = self.extract_sensor_readings(node_features)
        
        node_logits, no_leak_logit, pred_magnitude = self.model(sensor_readings)
        
        # Node loss
        node_loss = self.focal_loss(node_logits, y_node)
        
        # No-leak loss
        no_leak_target = 1.0 - has_leak
        no_leak_loss = F.binary_cross_entropy_with_logits(
            no_leak_logit.squeeze(-1), no_leak_target
        )
        
        # Magnitude loss (only for samples with leaks)
        leak_mask = has_leak > 0.5
        if leak_mask.sum() > 0:
            magnitude_loss = F.mse_loss(
                pred_magnitude[leak_mask].squeeze(-1),
                leak_magnitude[leak_mask]
            )
        else:
            magnitude_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = node_loss + 0.5 * no_leak_loss + 0.3 * magnitude_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "node_loss": node_loss.item(),
            "no_leak_loss": no_leak_loss.item(),
            "magnitude_loss": magnitude_loss.item(),
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        batch: dict,
    ) -> dict:
        """
        Evaluate on a batch.
        
        Args:
            batch: Batch of data.
        
        Returns:
            Dictionary with predictions and targets.
        """
        self.model.eval()
        
        node_features = batch["node_features"].to(self.device)
        sensor_readings = self.extract_sensor_readings(node_features)
        
        node_probs, no_leak_prob, pred_magnitude = self.model.predict(sensor_readings)
        
        return {
            "node_probs": node_probs.cpu().numpy(),
            "no_leak_prob": no_leak_prob.cpu().numpy(),
            "pred_magnitude": pred_magnitude.cpu().numpy(),
        }


class FocalLossSimple(nn.Module):
    """Simple focal loss implementation."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * ce_loss).mean()
