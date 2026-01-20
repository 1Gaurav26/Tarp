"""
Temporal-Aware Leak Detection with Time-Series Graph Neural Networks.

PATENTABLE INNOVATION: Integration of temporal patterns using time-series
graph neural networks for leak evolution tracking and early warning.

Key Features:
- Temporal GNN with attention mechanisms over time windows
- Leak growth rate estimation
- Early warning system with trend analysis
- Anomaly detection using temporal residuals
"""

from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_stage1 import LeakDetectionGNN, MessagePassingLayer


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for aggregating time-series information.
    
    Learns to weight different time steps based on their relevance for
    leak detection and evolution tracking.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, num_nodes, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Temporal sequence of node embeddings.
        
        Returns:
            Tuple of (attended_embeddings, attention_weights)
        """
        batch_size, seq_len, num_nodes, hidden_dim = x.shape
        
        # Reshape: [batch * num_nodes, seq_len, hidden_dim]
        x_reshaped = x.transpose(1, 2).contiguous()
        x_reshaped = x_reshaped.view(batch_size * num_nodes, seq_len, hidden_dim)
        
        # Compute Q, K, V
        Q = self.q_proj(x_reshaped)  # [batch * num_nodes, seq_len, hidden_dim]
        K = self.k_proj(x_reshaped)
        V = self.v_proj(x_reshaped)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size * num_nodes, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # [batch * num_nodes, num_heads, seq_len, seq_len]
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch * num_nodes, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size * num_nodes, seq_len, hidden_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual connection
        output = output + x_reshaped
        output = self.norm(output)
        
        # Reshape back: [batch, num_nodes, seq_len, hidden_dim]
        output = output.view(batch_size, num_nodes, seq_len, hidden_dim)
        output = output.transpose(1, 2)  # [batch, seq_len, num_nodes, hidden_dim]
        
        # Average attention weights across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=1)  # [batch * num_nodes, seq_len, seq_len]
        attn_weights_avg = attn_weights_avg.view(batch_size, num_nodes, seq_len, seq_len)
        
        return output, attn_weights_avg


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for leak detection with time-series awareness.
    
    PATENTABLE FEATURE: Processes temporal sequences of sensor readings using
    graph-based temporal attention for leak evolution tracking.
    """
    
    def __init__(
        self,
        base_gnn: Optional[LeakDetectionGNN] = None,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        temporal_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize temporal GNN.
        
        Args:
            base_gnn: Pre-trained base GNN (optional).
            in_dim: Input node feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of message passing layers.
            edge_dim: Edge feature dimension.
            temporal_heads: Number of temporal attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Base GNN for spatial processing at each time step
        if base_gnn is not None:
            self.base_gnn = base_gnn
        else:
            self.base_gnn = LeakDetectionGNN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=edge_dim,
                dropout=dropout,
            )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=temporal_heads,
        )
        
        # Leak evolution head: predicts leak growth rate
        self.evolution_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Current + trend
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Growth rate
        )
        
        # Trend extraction: simple linear regression over time window
        self.trend_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x_seq: torch.Tensor,  # [batch, seq_len, num_nodes, in_dim]
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with temporal sequence.
        
        Args:
            x_seq: Temporal sequence of node features.
            edge_index: Edge indices (same across time).
            edge_attr: Edge attributes (same across time).
        
        Returns:
            Tuple of:
            - node_probs: Current leak probabilities [batch, num_nodes]
            - growth_rates: Leak growth rates [batch, num_nodes]
            - temporal_embeddings: Temporal node embeddings [batch, seq_len, num_nodes, hidden_dim]
            - attention_weights: Temporal attention weights
        """
        batch_size, seq_len, num_nodes, in_dim = x_seq.shape
        
        # Process each time step with base GNN
        temporal_embeddings = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :, :]  # [batch, num_nodes, in_dim]
            
            # Get embeddings from base GNN
            h = self.base_gnn.input_proj(x_t)
            for mp_layer in self.base_gnn.mp_layers:
                h = mp_layer(h, edge_index, edge_attr)
            
            temporal_embeddings.append(h)
        
        # Stack: [batch, seq_len, num_nodes, hidden_dim]
        temporal_embeddings = torch.stack(temporal_embeddings, dim=1)
        
        # Apply temporal attention
        attended_embeddings, attention_weights = self.temporal_attention(temporal_embeddings)
        
        # Use latest time step for current predictions
        current_embeddings = attended_embeddings[:, -1, :, :]  # [batch, num_nodes, hidden_dim]
        
        # Predict current leak probabilities
        node_logits = self.base_gnn.node_head(current_embeddings).squeeze(-1)
        node_probs = torch.sigmoid(node_logits)
        
        # Extract trend (linear regression over sequence)
        # Use recent time steps to compute trend
        recent_embeddings = attended_embeddings[:, -min(5, seq_len):, :, :]  # [batch, recent, num_nodes, hidden_dim]
        trend_embeddings = self._compute_trend(recent_embeddings)
        
        # Combine current and trend for growth rate prediction
        current_trend = torch.cat([current_embeddings, trend_embeddings], dim=-1)  # [batch, num_nodes, 2*hidden_dim]
        growth_logits = self.evolution_head(current_trend).squeeze(-1)  # [batch, num_nodes]
        growth_rates = torch.tanh(growth_logits)  # [-1, 1] growth rate
        
        return node_probs, growth_rates, attended_embeddings, attention_weights
    
    def _compute_trend(
        self,
        embeddings: torch.Tensor,  # [batch, seq_len, num_nodes, hidden_dim]
    ) -> torch.Tensor:
        """
        Compute trend using linear regression over time dimension.
        
        Args:
            embeddings: Temporal embeddings.
        
        Returns:
            Trend embeddings [batch, num_nodes, hidden_dim]
        """
        batch_size, seq_len, num_nodes, hidden_dim = embeddings.shape
        
        # Simple approach: difference between last and first
        # More sophisticated: linear regression coefficients
        trend = embeddings[:, -1, :, :] - embeddings[:, 0, :, :]
        trend = self.trend_proj(trend)
        
        return trend
    
    def predict_with_evolution(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        forecast_steps: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict leak probabilities with future evolution forecast.
        
        PATENTABLE METHOD: Uses temporal patterns to forecast leak evolution.
        
        Args:
            x_seq: Historical sequence.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            forecast_steps: Number of future steps to forecast.
        
        Returns:
            Dictionary with current predictions and forecasts.
        """
        node_probs, growth_rates, embeddings, attention = self.forward(
            x_seq, edge_index, edge_attr
        )
        
        # Forecast future evolution
        # Simple extrapolation: prob_t+1 = prob_t * (1 + growth_rate)
        current_probs = node_probs  # [batch, num_nodes]
        forecasts = []
        
        for step in range(forecast_steps):
            # Exponential growth with saturation
            forecast_probs = torch.clamp(
                current_probs * (1.0 + growth_rates * 0.1),  # Small growth per step
                0.0, 1.0
            )
            forecasts.append(forecast_probs)
            current_probs = forecast_probs
        
        forecasts_tensor = torch.stack(forecasts, dim=1)  # [batch, forecast_steps, num_nodes]
        
        return {
            "current_probs": node_probs,
            "growth_rates": growth_rates,
            "forecast_probs": forecasts_tensor,
            "attention_weights": attention,
        }


class TemporalAnomalyDetector(nn.Module):
    """
    Anomaly detection using temporal residuals.
    
    Detects leaks by identifying anomalous patterns in temporal sequences
    that deviate from expected behavior.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        seq_len: int = 10,
    ):
        """
        Initialize temporal anomaly detector.
        
        Args:
            hidden_dim: Hidden dimension.
            seq_len: Sequence length.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # LSTM for temporal pattern learning
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        
        # Anomaly score predictor
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # hidden + prediction error
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        temporal_embeddings: torch.Tensor,  # [batch, seq_len, num_nodes, hidden_dim]
    ) -> torch.Tensor:
        """
        Detect anomalies in temporal sequence.
        
        Args:
            temporal_embeddings: Temporal node embeddings.
        
        Returns:
            Anomaly scores [batch, num_nodes]
        """
        batch_size, seq_len, num_nodes, hidden_dim = temporal_embeddings.shape
        
        # Process each node's time series
        all_anomaly_scores = []
        
        for node in range(num_nodes):
            node_sequence = temporal_embeddings[:, :, node, :]  # [batch, seq_len, hidden_dim]
            
            # Predict next step
            lstm_out, _ = self.lstm(node_sequence)  # [batch, seq_len, hidden_dim]
            predictions = lstm_out[:, :-1, :]  # [batch, seq_len-1, hidden_dim]
            actuals = node_sequence[:, 1:, :]  # [batch, seq_len-1, hidden_dim]
            
            # Prediction error
            errors = torch.norm(predictions - actuals, dim=-1)  # [batch, seq_len-1]
            mean_error = errors.mean(dim=1, keepdim=True)  # [batch, 1]
            
            # Latest hidden state
            latest_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim]
            
            # Anomaly score
            anomaly_input = torch.cat([latest_hidden, mean_error.expand(-1, hidden_dim)], dim=-1)
            anomaly_score = self.anomaly_head(anomaly_input).squeeze(-1)  # [batch]
            
            all_anomaly_scores.append(anomaly_score)
        
        # Stack: [batch, num_nodes]
        anomaly_scores = torch.stack(all_anomaly_scores, dim=1)
        
        return anomaly_scores

