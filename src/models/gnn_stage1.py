"""
Stage 1 GNN for Water Leak Detection and Localization (Production).

Enhanced architecture with:
- Multi-head attention for adaptive neighbor weighting
- Edge-gated message passing (conductance modulates messages)
- Skip connections with projection across all layers
- BatchNorm for stable training
- Deeper prediction heads
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGatedAttentionLayer(nn.Module):
    """
    Message passing layer with edge-gated multi-head attention.
    
    Each attention head computes:
        alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || e_ij]))
    Messages are gated by edge features (conductance) and
    combined via multi-head attention.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Multi-head projections
        self.W_q = nn.Linear(in_dim, out_dim, bias=False)
        self.W_k = nn.Linear(in_dim, out_dim, bias=False)
        self.W_v = nn.Linear(in_dim, out_dim, bias=False)
        
        # Edge gate: maps edge features to per-head gate values
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, num_heads * 2),
            nn.ReLU(),
            nn.Linear(num_heads * 2, num_heads),
            nn.Sigmoid(),
        )
        
        # Attention scoring
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * self.head_dim + edge_dim, self.head_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.head_dim, 1),
        )
        
        # Update: combine current embedding with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        
        # Skip connection projection (always project for flexibility)
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            batch_size = x.size(0)
            return torch.stack([self._forward_single(x[b], edge_index, edge_attr) for b in range(batch_size)])
        return self._forward_single(x, edge_index, edge_attr)
    
    def _forward_single(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Project to queries, keys, values → [N, num_heads, head_dim]
        Q = self.W_q(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Edge gate: [E, num_heads]
        gate = self.edge_gate(edge_attr)
        
        # For each edge, compute attention scores per head
        num_edges = src.size(0)
        q_src = Q[src]  # [E, H, D]
        k_dst = K[dst]  # [E, H, D]
        
        # Attention score per head
        # expand edge_attr for each head: [E, edge_dim] → [E, H, edge_dim]
        edge_expanded = edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
        
        attn_input = torch.cat([q_src, k_dst, edge_expanded], dim=-1)  # [E, H, 2*D + edge_dim]
        attn_input_flat = attn_input.view(num_edges * self.num_heads, -1)
        attn_scores = self.attn_mlp(attn_input_flat).view(num_edges, self.num_heads)  # [E, H]
        
        # Apply edge gating
        attn_scores = attn_scores * gate  # [E, H]
        
        # Softmax per destination node per head using scatter
        # We need to compute softmax grouped by dst node
        attn_weights = self._scatter_softmax(attn_scores, dst, num_nodes)  # [E, H]
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate: weighted sum of values
        v_src = V[src]  # [E, H, D]
        weighted_v = v_src * attn_weights.unsqueeze(-1)  # [E, H, D]
        
        # Scatter add to destination nodes
        agg = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        dst_expanded = dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.head_dim)
        agg.scatter_add_(0, dst_expanded, weighted_v)
        
        # Reshape multi-head output: [N, out_dim]
        agg_flat = agg.view(num_nodes, self.out_dim)
        
        # Update with residual
        update_input = torch.cat([x, agg_flat], dim=-1)
        updated = self.update_mlp(update_input)
        
        # Skip connection + layer norm
        skip = self.skip_proj(x)
        out = self.norm(updated + skip)
        
        return out
    
    def _scatter_softmax(self, scores: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute softmax over scores grouped by index (destination nodes)."""
        # scores: [E, H], index: [E]
        # Subtract max for numerical stability
        max_vals = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        idx_exp = index.unsqueeze(1).expand_as(scores)
        max_vals.scatter_reduce_(0, idx_exp, scores, reduce='amax', include_self=False)
        max_vals = max_vals.clamp(min=-1e9)
        
        scores_shifted = scores - max_vals[index]
        exp_scores = torch.exp(scores_shifted)
        
        # Sum per node
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(0, idx_exp, exp_scores)
        
        return exp_scores / (sum_exp[index] + 1e-10)


class LeakDetectionGNN(nn.Module):
    """
    Production Graph Neural Network for leak detection and localization.
    
    Architecture:
    - Input projection with BatchNorm
    - Multiple edge-gated attention layers with skip connections
    - Per-node leak probability head (deeper MLP)
    - Global no-leak score via attention pooling
    """
    
    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        edge_dim: int = 2,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Edge-gated attention layers
        self.mp_layers = nn.ModuleList([
            EdgeGatedAttentionLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Per-layer dropout
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Per-node leak probability head (deeper)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Attention pooling for global representation
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Global no-leak classifier (deeper)
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
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
            sensor_mask: Optional sensor mask (unused in forward)
        
        Returns:
            node_logits: Per-node leak logits
            no_leak_logit: Global no-leak logit
        """
        batched = x.dim() == 3
        
        # Input projection (BatchNorm needs 2D input)
        if batched:
            B, N, D = x.shape
            h = self.input_proj(x.view(B * N, D)).view(B, N, self.hidden_dim)
        else:
            h = self.input_proj(x)
        
        # Message passing layers
        for mp_layer, dropout in zip(self.mp_layers, self.layer_dropouts):
            h = mp_layer(h, edge_index, edge_attr)
            h = dropout(h)
        
        # Per-node predictions
        node_logits = self.node_head(h).squeeze(-1)
        
        # Attention pooling for global feature
        if batched:
            # h: [B, N, H]
            attn_weights = self.pool_attn(h)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            h_global = (h * attn_weights).sum(dim=1)  # [B, H]
        else:
            attn_weights = self.pool_attn(h)  # [N, 1]
            attn_weights = F.softmax(attn_weights, dim=0)
            h_global = (h * attn_weights).sum(dim=0, keepdim=True)  # [1, H]
        
        no_leak_logit = self.global_head(h_global).squeeze(-1)
        
        return node_logits, no_leak_logit
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions as probabilities."""
        node_logits, no_leak_logit = self.forward(x, edge_index, edge_attr)
        node_probs = torch.sigmoid(node_logits)
        no_leak_prob = torch.sigmoid(no_leak_logit)
        return node_probs, no_leak_prob


class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    alpha controls the weight for positive class (leak nodes).
    For 1:199 class ratio, alpha should be HIGH (e.g. 0.9) to upweight
    the rare positive class.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for POSITIVE class
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        # alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss.mean()


def smooth_labels(y_node: torch.Tensor, edge_index: torch.Tensor, smooth_val: float = 0.3) -> torch.Tensor:
    """
    Neighborhood label smoothing: spread partial labels to 1-hop neighbors of leak nodes.
    
    This gives the GNN a gradient signal from nearby nodes, not just one node in 200.
    """
    smoothed = y_node.clone()
    if y_node.dim() == 2:
        # Batched: [B, N]
        src, dst = edge_index[0], edge_index[1]
        for b in range(y_node.size(0)):
            leak_mask = y_node[b] > 0.5
            if leak_mask.any():
                leak_nodes = leak_mask.nonzero(as_tuple=True)[0]
                for ln in leak_nodes:
                    # Find neighbors via edge_index
                    neighbors = dst[src == ln]
                    for nb in neighbors:
                        if smoothed[b, nb] < smooth_val:
                            smoothed[b, nb] = smooth_val
    else:
        # Single: [N]
        src, dst = edge_index[0], edge_index[1]
        leak_mask = y_node > 0.5
        if leak_mask.any():
            leak_nodes = leak_mask.nonzero(as_tuple=True)[0]
            for ln in leak_nodes:
                neighbors = dst[src == ln]
                for nb in neighbors:
                    if smoothed[nb] < smooth_val:
                        smoothed[nb] = smooth_val
    return smoothed


def compute_stage1_loss(
    node_logits: torch.Tensor,
    no_leak_logit: torch.Tensor,
    y_node: torch.Tensor,
    has_leak: torch.Tensor,
    focal_gamma: float = 2.0,
    no_leak_weight: float = 0.5,
    edge_index: torch.Tensor = None,
    smooth_val: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined Stage 1 loss with neighborhood smoothing.
    
    Loss = FocalLoss(node_logits, smoothed_y_node) + no_leak_weight * BCE(no_leak, 1-has_leak)
    """
    # Apply neighborhood label smoothing if edge_index provided
    if edge_index is not None:
        y_smooth = smooth_labels(y_node, edge_index, smooth_val)
    else:
        y_smooth = y_node
    
    # Focal loss with alpha=0.9 (upweight positive/leak class)
    focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=0.9)
    node_loss = focal_loss_fn(node_logits, y_smooth)
    
    no_leak_target = 1.0 - has_leak
    no_leak_loss = F.binary_cross_entropy_with_logits(no_leak_logit, no_leak_target)
    
    total_loss = node_loss + no_leak_weight * no_leak_loss
    
    loss_dict = {
        "node_loss": node_loss.item(),
        "no_leak_loss": no_leak_loss.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict
