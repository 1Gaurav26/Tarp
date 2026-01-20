"""
Transfer Learning Framework for Cross-Network Adaptation.

PATENTABLE INNOVATION: Domain adaptation techniques for transferring leak
detection models across different water network topologies.

Key Features:
- Graph embedding alignment across networks
- Domain adversarial training for generalization
- Few-shot learning for new network configurations
- Meta-learning for rapid adaptation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from .gnn_stage1 import LeakDetectionGNN, MessagePassingLayer


class GraphAdapter(nn.Module):
    """
    Graph structure adapter for cross-network transfer.
    
    Adapts node and edge features to align across different network topologies.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        adapter_dim: int = 32,
    ):
        """
        Initialize graph adapter.
        
        Args:
            hidden_dim: Hidden dimension.
            adapter_dim: Adapter dimension (smaller for domain-specific adaptation).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        
        # Adapter layers (bottleneck for domain adaptation)
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, num_nodes, hidden_dim]
    ) -> torch.Tensor:
        """
        Adapt features for target domain.
        
        Args:
            x: Source domain features.
        
        Returns:
            Adapted features [batch, num_nodes, hidden_dim]
        """
        # Bottleneck adaptation
        adapted = self.down_proj(x)
        adapted = F.relu(adapted)
        adapted = self.up_proj(adapted)
        
        # Residual connection
        adapted = adapted + x
        adapted = self.norm(adapted)
        
        return adapted


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial training.
    
    Tries to distinguish between source and target domains.
    Used in adversarial training to learn domain-invariant features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        """
        Initialize domain discriminator.
        
        Args:
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),  # Binary: source vs target
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # [batch, num_nodes, hidden_dim]
    ) -> torch.Tensor:
        """
        Predict domain (source vs target).
        
        Args:
            node_embeddings: Node embeddings.
        
        Returns:
            Domain probabilities [batch, num_nodes]
        """
        # Global pooling
        if node_embeddings.dim() == 3:
            h_global = node_embeddings.mean(dim=1)  # [batch, hidden_dim]
        else:
            h_global = node_embeddings.mean(dim=0, keepdim=True)
        
        domain_prob = self.discriminator(h_global).squeeze(-1)
        
        return domain_prob


class GraphEmbeddingAlignment(nn.Module):
    """
    Aligns graph embeddings across different network topologies.
    
    Uses optimal transport or attention mechanisms to align nodes
    across source and target networks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        """
        Initialize embedding alignment module.
        
        Args:
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Alignment network
        self.alignment_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def compute_alignment(
        self,
        source_embeddings: torch.Tensor,  # [batch, num_nodes_src, hidden_dim]
        target_embeddings: torch.Tensor,  # [batch, num_nodes_tgt, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment between source and target embeddings.
        
        Uses attention mechanism to find correspondences.
        
        Args:
            source_embeddings: Source network embeddings.
            target_embeddings: Target network embeddings.
        
        Returns:
            Tuple of (aligned_source_embeddings, alignment_weights)
        """
        batch_size = source_embeddings.shape[0]
        num_src = source_embeddings.shape[1]
        num_tgt = target_embeddings.shape[1]
        
        # Compute similarity matrix
        # [batch, num_src, hidden_dim] @ [batch, hidden_dim, num_tgt]
        similarity = torch.matmul(
            source_embeddings,
            target_embeddings.transpose(-2, -1),
        ) / np.sqrt(self.hidden_dim)
        
        # Soft alignment
        alignment_weights = F.softmax(similarity, dim=-1)  # [batch, num_src, num_tgt]
        
        # Aligned source embeddings
        aligned_source = torch.matmul(alignment_weights, target_embeddings)
        # [batch, num_src, hidden_dim]
        
        # Combine with original
        combined = torch.cat([source_embeddings, aligned_source], dim=-1)
        aligned = self.alignment_net(combined)
        
        return aligned, alignment_weights


class TransferableLeakDetectionGNN(nn.Module):
    """
    Transferable GNN for cross-network leak detection.
    
    PATENTABLE FEATURE: Domain adaptation framework that enables
    transfer of leak detection models across different network topologies.
    """
    
    def __init__(
        self,
        base_gnn: Optional[LeakDetectionGNN] = None,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        use_adapter: bool = True,
        use_adversarial: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize transferable GNN.
        
        Args:
            base_gnn: Pre-trained base GNN (optional).
            in_dim: Input node feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of message passing layers.
            edge_dim: Edge feature dimension.
            use_adapter: Whether to use adapter layers.
            use_adversarial: Whether to use adversarial training.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_adapter = use_adapter
        self.use_adversarial = use_adversarial
        self.dropout = dropout
        
        # Base GNN
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
        
        # Graph adapter (for domain adaptation)
        if use_adapter:
            self.adapter = GraphAdapter(hidden_dim, adapter_dim=hidden_dim // 2)
        else:
            self.adapter = None
        
        # Domain discriminator (for adversarial training)
        if use_adversarial:
            self.domain_discriminator = DomainDiscriminator(hidden_dim)
        else:
            self.domain_discriminator = None
        
        # Embedding alignment
        self.embedding_alignment = GraphEmbeddingAlignment(hidden_dim)
        
        # Domain-specific prediction heads (optional)
        self.domain_head_source = self.base_gnn.node_head  # Use base head for source
        self.domain_head_target = nn.Sequential(
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
        domain: str = "source",  # "source" or "target"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with domain adaptation.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            domain: Domain identifier ("source" or "target").
        
        Returns:
            Tuple of:
            - node_logits: Per-node leak logits [batch, num_nodes]
            - no_leak_logit: Global no-leak logit [batch]
            - domain_prob: Domain prediction [batch] (if adversarial)
        """
        # Base GNN forward (up to embeddings)
        h = self.base_gnn.input_proj(x)
        for mp_layer in self.base_gnn.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Apply adapter if target domain
        if self.use_adapter and domain == "target":
            h = self.adapter(h)
        
        # Domain discriminator (for adversarial training)
        domain_prob = None
        if self.use_adversarial and self.domain_discriminator is not None:
            domain_prob = self.domain_discriminator(h)
        
        # Prediction (domain-specific head for target)
        if domain == "target" and self.use_adapter:
            node_logits = self.domain_head_target(h).squeeze(-1)
        else:
            node_logits = self.base_gnn.node_head(h).squeeze(-1)
        
        # Global pooling
        if h.dim() == 3:
            h_global = h.mean(dim=1)
        else:
            h_global = h.mean(dim=0, keepdim=True)
        
        no_leak_logit = self.base_gnn.global_head(h_global).squeeze(-1)
        
        return node_logits, no_leak_logit, domain_prob
    
    def align_embeddings(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align embeddings between source and target networks.
        
        Args:
            source_embeddings: Source network embeddings.
            target_embeddings: Target network embeddings.
        
        Returns:
            Tuple of (aligned_embeddings, alignment_weights)
        """
        return self.embedding_alignment.compute_alignment(
            source_embeddings, target_embeddings
        )
    
    def fine_tune_on_target(
        self,
        target_data: List[Dict],
        epochs: int = 10,
        lr: float = 1e-4,
        freeze_base: bool = True,
    ):
        """
        Fine-tune model on target domain with few samples.
        
        PATENTABLE METHOD: Few-shot adaptation using adapter layers
        and alignment mechanisms.
        
        Args:
            target_data: List of target domain samples.
            epochs: Number of fine-tuning epochs.
            lr: Learning rate.
            freeze_base: Whether to freeze base GNN weights.
        """
        if freeze_base:
            # Freeze base GNN
            for param in self.base_gnn.parameters():
                param.requires_grad = False
            
            # Only train adapter and target head
            trainable_params = list(self.adapter.parameters()) + \
                             list(self.domain_head_target.parameters())
        else:
            trainable_params = self.parameters()
        
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            
            for sample in target_data:
                x = sample["node_features"]
                edge_index = sample["edge_index"]
                edge_attr = sample["edge_attr"]
                y_node = sample["y_node"]
                
                optimizer.zero_grad()
                
                node_logits, _, _ = self.forward(
                    x, edge_index, edge_attr, domain="target"
                )
                
                loss = criterion(node_logits, y_node)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(target_data)
            print(f"Fine-tuning epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def compute_transfer_loss(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    target_logits: torch.Tensor,
    target_labels: torch.Tensor,
    domain_probs: torch.Tensor,
    domain_labels: torch.Tensor,
    lambda_adv: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute transfer learning loss with adversarial component.
    
    Combines:
    - Task loss (source + target)
    - Adversarial loss (domain confusion)
    
    Args:
        source_logits: Source domain logits [batch_src, num_nodes]
        source_labels: Source domain labels [batch_src, num_nodes]
        target_logits: Target domain logits [batch_tgt, num_nodes]
        target_labels: Target domain labels [batch_tgt, num_nodes]
        domain_probs: Domain predictions [batch_total]
        domain_labels: Domain labels (0=source, 1=target) [batch_total]
        lambda_adv: Adversarial loss weight.
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Task loss (source)
    source_loss = F.binary_cross_entropy_with_logits(source_logits, source_labels)
    
    # Task loss (target, if labels available)
    if target_labels is not None:
        target_loss = F.binary_cross_entropy_with_logits(target_logits, target_labels)
    else:
        target_loss = torch.tensor(0.0, device=source_logits.device)
    
    # Adversarial loss (domain confusion)
    # We want discriminator to fail (gradient reversal)
    domain_loss = F.binary_cross_entropy_with_logits(
        domain_probs,
        domain_labels,
    )
    
    # Total loss
    total_loss = source_loss + target_loss - lambda_adv * domain_loss
    
    loss_dict = {
        "source_loss": source_loss.item(),
        "target_loss": target_loss.item() if target_labels is not None else 0.0,
        "domain_loss": domain_loss.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict

