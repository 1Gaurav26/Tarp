"""
Multi-Modal Sensor Fusion Architecture for Leak Detection.

PATENTABLE INNOVATION: Integration of pressure, flow, and acoustic sensors
using attention-based fusion in graph neural networks.

Key Features:
- Cross-modal attention mechanisms
- Feature-level and decision-level fusion
- Missing sensor handling with learned imputation
- Modal importance weighting
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_stage1 import LeakDetectionGNN, MessagePassingLayer


class ModalEncoder(nn.Module):
    """
    Encoder for individual sensor modalities.
    
    Each modality (pressure, flow, acoustic) has its own encoder
    that processes sensor-specific features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        modal_name: str = "pressure",
    ):
        """
        Initialize modal encoder.
        
        Args:
            input_dim: Input feature dimension for this modality.
            hidden_dim: Hidden dimension.
            modal_name: Name of modality (for identification).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modal_name = modal_name
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # Modal embedding (learnable per modality)
        self.modal_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.1)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, num_nodes, input_dim]
    ) -> torch.Tensor:
        """
        Encode modal features.
        
        Args:
            x: Modal-specific features.
        
        Returns:
            Encoded features [batch, num_nodes, hidden_dim]
        """
        encoded = self.encoder(x)
        
        # Add modal embedding
        batch_size = encoded.shape[0]
        modal_emb = self.modal_embedding.expand(batch_size, -1, -1)
        # Broadcast modal embedding to all nodes
        encoded = encoded + modal_emb.mean(dim=1, keepdim=True)
        
        return encoded


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.
    
    Allows modalities to attend to each other for complementary information.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query_modal: torch.Tensor,  # [batch, num_nodes, hidden_dim]
        key_modal: torch.Tensor,    # [batch, num_nodes, hidden_dim]
        value_modal: torch.Tensor,  # [batch, num_nodes, hidden_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            query_modal: Query from one modality.
            key_modal: Keys from another modality.
            value_modal: Values from another modality.
        
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, num_nodes, hidden_dim = query_modal.shape
        
        # Reshape for multi-head: [batch, num_nodes, num_heads, head_dim]
        Q = self.q_proj(query_modal).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(key_modal).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(value_modal).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)  # [batch, num_heads, num_nodes, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn_weights, V)  # [batch, num_heads, num_nodes, head_dim]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, num_nodes, hidden_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        # Residual
        output = output + query_modal
        output = self.norm(output)
        
        return output, attn_weights.mean(dim=1)  # Average attention across heads


class MissingSensorImputation(nn.Module):
    """
    Learned imputation for missing sensor readings.
    
    Uses graph structure and available sensors to impute missing values.
    """
    
    def __init__(
        self,
        hidden_dim: int,
    ):
        """
        Initialize imputation model.
        
        Args:
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Imputation network
        self.imputation_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Available + context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        node_embeddings: torch.Tensor,  # [batch, num_nodes, hidden_dim]
        missing_mask: torch.Tensor,     # [batch, num_nodes] binary mask (1 = missing)
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Impute missing sensor readings.
        
        Args:
            node_embeddings: Node embeddings.
            missing_mask: Missing sensor mask.
            edge_index: Edge indices for graph.
        
        Returns:
            Imputed embeddings [batch, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        
        # Aggregate neighbor features for imputation
        neighbor_embeddings = torch.zeros_like(node_embeddings)
        
        src, dst = edge_index[0], edge_index[1]
        for i in range(batch_size):
            for j in range(len(dst)):
                d = dst[j].item()
                s = src[j].item()
                neighbor_embeddings[i, d] += node_embeddings[i, s]
        
        # Count neighbors (simplified)
        neighbor_counts = torch.ones(num_nodes, device=node_embeddings.device)
        neighbor_embeddings = neighbor_embeddings / neighbor_counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        
        # Impute missing
        imputed_embeddings = node_embeddings.clone()
        
        # For missing nodes, use neighbor context
        missing_indices = missing_mask.bool()
        if missing_indices.any():
            context_features = torch.cat([
                neighbor_embeddings,
                node_embeddings,
            ], dim=-1)
            
            imputed = self.imputation_mlp(context_features)
            
            # Replace missing with imputed
            imputed_embeddings[missing_indices] = imputed[missing_indices]
        
        return imputed_embeddings


class MultiModalFusionGNN(nn.Module):
    """
    Multi-modal fusion Graph Neural Network for leak detection.
    
    PATENTABLE FEATURE: Integrates pressure, flow, and acoustic sensors
    using attention-based fusion with missing sensor handling.
    """
    
    def __init__(
        self,
        pressure_dim: int = 2,
        flow_dim: int = 1,
        acoustic_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        num_modals: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-modal fusion GNN.
        
        Args:
            pressure_dim: Pressure sensor feature dimension.
            flow_dim: Flow sensor feature dimension.
            acoustic_dim: Acoustic sensor feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of message passing layers.
            edge_dim: Edge feature dimension.
            num_modals: Number of modalities.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modals = num_modals
        self.dropout = dropout
        
        # Modal encoders
        self.pressure_encoder = ModalEncoder(pressure_dim, hidden_dim, "pressure")
        self.flow_encoder = ModalEncoder(flow_dim, hidden_dim, "flow")
        self.acoustic_encoder = ModalEncoder(acoustic_dim, hidden_dim, "acoustic")
        
        # Cross-modal attention
        self.cross_modal_attn = nn.ModuleDict({
            "pressure_to_flow": CrossModalAttention(hidden_dim),
            "pressure_to_acoustic": CrossModalAttention(hidden_dim),
            "flow_to_pressure": CrossModalAttention(hidden_dim),
            "flow_to_acoustic": CrossModalAttention(hidden_dim),
            "acoustic_to_pressure": CrossModalAttention(hidden_dim),
            "acoustic_to_flow": CrossModalAttention(hidden_dim),
        })
        
        # Missing sensor imputation
        self.imputation = MissingSensorImputation(hidden_dim)
        
        # Modal importance weighting
        self.modal_importance = nn.Parameter(torch.ones(num_modals) / num_modals)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_modals, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Message passing layers (after fusion)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        
        # Prediction heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        pressure_features: torch.Tensor,      # [batch, num_nodes, pressure_dim]
        flow_features: torch.Tensor,          # [batch, num_nodes, flow_dim]
        acoustic_features: torch.Tensor,      # [batch, num_nodes, acoustic_dim]
        pressure_mask: Optional[torch.Tensor] = None,  # [batch, num_nodes] (1 = available)
        flow_mask: Optional[torch.Tensor] = None,
        acoustic_mask: Optional[torch.Tensor] = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-modal fusion.
        
        Args:
            pressure_features: Pressure sensor features.
            flow_features: Flow sensor features.
            acoustic_features: Acoustic sensor features.
            pressure_mask: Availability mask for pressure sensors.
            flow_mask: Availability mask for flow sensors.
            acoustic_mask: Availability mask for acoustic sensors.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
        
        Returns:
            Tuple of:
            - node_logits: Per-node leak logits [batch, num_nodes]
            - no_leak_logit: Global no-leak logit [batch]
            - attention_dict: Cross-modal attention weights
        """
        batch_size, num_nodes, _ = pressure_features.shape
        
        # Encode each modality
        pressure_emb = self.pressure_encoder(pressure_features)
        flow_emb = self.flow_encoder(flow_features)
        acoustic_emb = self.acoustic_encoder(acoustic_features)
        
        # Handle missing sensors with imputation
        if pressure_mask is not None:
            missing_pressure = 1 - pressure_mask
            pressure_emb = self.imputation(pressure_emb, missing_pressure, edge_index)
        
        if flow_mask is not None:
            missing_flow = 1 - flow_mask
            flow_emb = self.imputation(flow_emb, missing_flow, edge_index)
        
        if acoustic_mask is not None:
            missing_acoustic = 1 - acoustic_mask
            acoustic_emb = self.imputation(acoustic_emb, missing_acoustic, edge_index)
        
        # Cross-modal attention
        attention_dict = {}
        
        # Pressure attends to flow and acoustic
        pressure_flow, attn_pf = self.cross_modal_attn["pressure_to_flow"](
            pressure_emb, flow_emb, flow_emb
        )
        pressure_acoustic, attn_pa = self.cross_modal_attn["pressure_to_acoustic"](
            pressure_emb, acoustic_emb, acoustic_emb
        )
        attention_dict["pressure_to_flow"] = attn_pf
        attention_dict["pressure_to_acoustic"] = attn_pa
        
        # Flow attends to pressure and acoustic
        flow_pressure, attn_fp = self.cross_modal_attn["flow_to_pressure"](
            flow_emb, pressure_emb, pressure_emb
        )
        flow_acoustic, attn_fa = self.cross_modal_attn["flow_to_acoustic"](
            flow_emb, acoustic_emb, acoustic_emb
        )
        attention_dict["flow_to_pressure"] = attn_fp
        attention_dict["flow_to_acoustic"] = attn_fa
        
        # Acoustic attends to pressure and flow
        acoustic_pressure, attn_ap = self.cross_modal_attn["acoustic_to_pressure"](
            acoustic_emb, pressure_emb, pressure_emb
        )
        acoustic_flow, attn_af = self.cross_modal_attn["acoustic_to_flow"](
            acoustic_emb, flow_emb, flow_emb
        )
        attention_dict["acoustic_to_pressure"] = attn_ap
        attention_dict["acoustic_to_flow"] = attn_af
        
        # Weighted combination of attended features
        # Use learnable modal importance weights
        modal_weights = F.softmax(self.modal_importance, dim=0)
        
        # Combine modalities
        combined_pressure = modal_weights[0] * (pressure_emb + pressure_flow + pressure_acoustic) / 3
        combined_flow = modal_weights[1] * (flow_emb + flow_pressure + flow_acoustic) / 3
        combined_acoustic = modal_weights[2] * (acoustic_emb + acoustic_pressure + acoustic_flow) / 3
        
        # Concatenate and fuse
        fused_features = torch.cat([
            combined_pressure,
            combined_flow,
            combined_acoustic,
        ], dim=-1)  # [batch, num_nodes, hidden_dim * 3]
        
        fused = self.fusion_layer(fused_features)  # [batch, num_nodes, hidden_dim]
        
        # Message passing on fused features
        h = fused
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Predictions
        node_logits = self.node_head(h).squeeze(-1)  # [batch, num_nodes]
        
        # Global pooling
        h_global = h.mean(dim=1)  # [batch, hidden_dim]
        no_leak_logit = self.global_head(h_global).squeeze(-1)  # [batch]
        
        return node_logits, no_leak_logit, attention_dict
    
    def predict(
        self,
        pressure_features: torch.Tensor,
        flow_features: torch.Tensor,
        acoustic_features: torch.Tensor,
        pressure_mask: Optional[torch.Tensor] = None,
        flow_mask: Optional[torch.Tensor] = None,
        acoustic_mask: Optional[torch.Tensor] = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions (probabilities).
        
        Returns:
            Tuple of (node_probs, no_leak_prob)
        """
        node_logits, no_leak_logit, _ = self.forward(
            pressure_features, flow_features, acoustic_features,
            pressure_mask, flow_mask, acoustic_mask,
            edge_index, edge_attr,
        )
        
        node_probs = torch.sigmoid(node_logits)
        no_leak_prob = torch.sigmoid(no_leak_logit)
        
        return node_probs, no_leak_prob

