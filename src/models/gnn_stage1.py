"""
Stage 1 GNN for Water Leak Detection and Localization.

Implements a Graph Neural Network that:
- Takes node features (sensor readings, masks) and edge attributes
- Outputs per-node leak probability and global no-leak score
- Uses manual message passing (no PyTorch Geometric dependency)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassingLayer(nn.Module):
    """
    Manual implementation of a message passing layer.
    
    Aggregates messages from neighbors and updates node representations.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 2,
        aggr: str = "mean",
    ):
        """
        Initialize the message passing layer.
        
        Args:
            in_dim: Input node feature dimension.
            out_dim: Output node feature dimension.
            edge_dim: Edge feature dimension.
            aggr: Aggregation method ("mean", "sum", "max").
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.aggr = aggr
        
        # Message MLP: combines source, target, and edge features
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        
        # Update MLP: combines current node with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of message passing.
        
        Args:
            x: Node features [num_nodes, in_dim] or [batch, num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_dim] or [batch, num_nodes, out_dim]
        """
        # Handle batched input
        if x.dim() == 3:
            batch_size, num_nodes, _ = x.shape
            outputs = []
            for b in range(batch_size):
                out = self._forward_single(x[b], edge_index, edge_attr)
                outputs.append(out)
            return torch.stack(outputs)
        else:
            return self._forward_single(x, edge_index, edge_attr)
    
    def _forward_single(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for single graph."""
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Compute messages
        src_features = x[src]  # [num_edges, in_dim]
        dst_features = x[dst]  # [num_edges, in_dim]
        
        # Concatenate source, target, and edge features
        message_input = torch.cat([src_features, dst_features, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)  # [num_edges, out_dim]
        
        # Aggregate messages per node
        aggregated = torch.zeros(num_nodes, self.out_dim, device=x.device)
        
        if self.aggr == "sum":
            aggregated.index_add_(0, dst, messages)
        elif self.aggr == "mean":
            # Sum and count
            aggregated.index_add_(0, dst, messages)
            count = torch.zeros(num_nodes, device=x.device)
            count.index_add_(0, dst, torch.ones(len(dst), device=x.device))
            count = count.clamp(min=1).unsqueeze(-1)
            aggregated = aggregated / count
        elif self.aggr == "max":
            # Use scatter_max equivalent
            for i, (d, m) in enumerate(zip(dst, messages)):
                aggregated[d] = torch.max(aggregated[d], m)
        
        # Update node features
        update_input = torch.cat([x, aggregated], dim=-1)
        updated = self.update_mlp(update_input)
        
        # Residual connection (if dimensions match)
        if self.in_dim == self.out_dim:
            updated = updated + x
        
        return self.norm(updated)


class LeakDetectionGNN(nn.Module):
    """
    Graph Neural Network for leak detection and localization.
    
    Architecture:
    - Input projection
    - Multiple message passing layers
    - Per-node leak probability prediction
    - Global no-leak score (via pooling)
    """
    
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize the GNN.
        
        Args:
            in_dim: Input node feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of message passing layers.
            edge_dim: Edge feature dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        
        # Per-node leak probability head
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Global no-leak classifier
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [batch, num_nodes, in_dim] or [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            sensor_mask: Optional sensor mask (not used in forward, but available)
        
        Returns:
            Tuple of:
            - node_logits: Per-node leak logits [batch, num_nodes] or [num_nodes]
            - no_leak_logit: Global no-leak logit [batch] or scalar
        """
        # Input projection
        h = self.input_proj(x)
        
        # Message passing
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Per-node predictions
        node_logits = self.node_head(h).squeeze(-1)  # [batch, num_nodes] or [num_nodes]
        
        # Global pooling for no-leak prediction
        if h.dim() == 3:
            # Batched: [batch, num_nodes, hidden]
            h_global = h.mean(dim=1)  # [batch, hidden]
        else:
            h_global = h.mean(dim=0, keepdim=True)  # [1, hidden]
        
        no_leak_logit = self.global_head(h_global).squeeze(-1)  # [batch] or [1]
        
        return node_logits, no_leak_logit
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions (probabilities instead of logits).
        
        Returns:
            Tuple of:
            - node_probs: Per-node leak probabilities
            - no_leak_prob: Probability of no leak
        """
        node_logits, no_leak_logit = self.forward(x, edge_index, edge_attr)
        node_probs = torch.sigmoid(node_logits)
        no_leak_prob = torch.sigmoid(no_leak_logit)
        return node_probs, no_leak_prob


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Initialize Focal Loss.
        
        Args:
            gamma: Focusing parameter. gamma=0 gives BCE.
            alpha: Weighting factor for positive class.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Predicted logits.
            targets: Binary targets.
        
        Returns:
            Focal loss value.
        """
        probs = torch.sigmoid(logits)
        
        # Compute cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


def compute_stage1_loss(
    node_logits: torch.Tensor,
    no_leak_logit: torch.Tensor,
    y_node: torch.Tensor,
    has_leak: torch.Tensor,
    focal_gamma: float = 2.0,
    no_leak_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss for Stage 1.
    
    Args:
        node_logits: Per-node leak logits [batch, num_nodes]
        no_leak_logit: No-leak logits [batch]
        y_node: Ground truth node labels [batch, num_nodes]
        has_leak: Whether each sample has a leak [batch]
        focal_gamma: Focal loss gamma parameter.
        no_leak_weight: Weight for no-leak loss component.
    
    Returns:
        Tuple of (total_loss, loss_dict with individual components)
    """
    # Node-level loss (focal loss)
    focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=0.25)
    node_loss = focal_loss_fn(node_logits, y_node)
    
    # Global no-leak loss (BCE)
    # Target: 1 if no leak, 0 if leak
    no_leak_target = 1.0 - has_leak
    no_leak_loss = F.binary_cross_entropy_with_logits(no_leak_logit, no_leak_target)
    
    # Combined loss
    total_loss = node_loss + no_leak_weight * no_leak_loss
    
    loss_dict = {
        "node_loss": node_loss.item(),
        "no_leak_loss": no_leak_loss.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict
