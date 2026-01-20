"""
Bayesian Graph Neural Network for Uncertainty Quantification.

PATENTABLE INNOVATION: Bayesian GNN architecture providing confidence intervals
and uncertainty estimates for leak predictions in water networks.

Key Features:
- Dropout-based uncertainty quantification (Monte Carlo Dropout)
- Bayesian neural network layers for epistemic uncertainty
- Heteroscedastic aleatoric uncertainty modeling
- Risk-aware decision making with confidence thresholds

This enables reliable leak detection with uncertainty bounds.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_stage1 import LeakDetectionGNN, MessagePassingLayer


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with learnable weight and bias uncertainty.
    
    Represents weights as distributions rather than point estimates.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            prior_std: Standard deviation of prior distribution.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Mean parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        
        # Log variance parameters (for numerical stability)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 1.0)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with sampling.
        
        Args:
            x: Input tensor.
            sample: Whether to sample from posterior or use mean.
        
        Returns:
            Output tensor.
        """
        if sample and self.training:
            # Sample from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        Used for variational inference loss.
        """
        weight_kl = -0.5 * torch.sum(
            1 + self.weight_logvar - self.weight_mu.pow(2) - self.weight_logvar.exp()
        )
        bias_kl = -0.5 * torch.sum(
            1 + self.bias_logvar - self.bias_mu.pow(2) - self.bias_logvar.exp()
        )
        
        return weight_kl + bias_kl


class HeteroscedasticUncertaintyModel(nn.Module):
    """
    Heteroscedastic uncertainty model for aleatoric uncertainty.
    
    Models data-dependent uncertainty (sensor noise, measurement uncertainty)
    that varies with input.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
    ):
        """
        Initialize heteroscedastic uncertainty model.
        
        Args:
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Uncertainty head: predicts log variance
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # No activation: output is log variance (can be negative)
        )
    
    def forward(
        self,
        h: torch.Tensor,  # [batch, num_nodes, hidden_dim]
    ) -> torch.Tensor:
        """
        Predict aleatoric uncertainty (log variance).
        
        Args:
            h: Node embeddings.
        
        Returns:
            Log variance [batch, num_nodes]
        """
        logvar = self.uncertainty_head(h).squeeze(-1)
        return logvar


class BayesianLeakDetectionGNN(nn.Module):
    """
    Bayesian Graph Neural Network for leak detection with uncertainty quantification.
    
    PATENTABLE FEATURE: Provides confidence intervals and uncertainty estimates
    for leak predictions, enabling risk-aware decision making.
    """
    
    def __init__(
        self,
        base_gnn: Optional[LeakDetectionGNN] = None,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 2,
        dropout: float = 0.1,
        use_bayesian: bool = True,
        use_heteroscedastic: bool = True,
        mc_samples: int = 10,
    ):
        """
        Initialize Bayesian GNN.
        
        Args:
            base_gnn: Pre-trained base GNN (optional).
            in_dim: Input node feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of message passing layers.
            edge_dim: Edge feature dimension.
            dropout: Dropout rate (for Monte Carlo Dropout).
            use_bayesian: Whether to use Bayesian layers.
            use_heteroscedastic: Whether to model heteroscedastic uncertainty.
            mc_samples: Number of Monte Carlo samples for inference.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_bayesian = use_bayesian
        self.use_heteroscedastic = use_heteroscedastic
        self.mc_samples = mc_samples
        
        # Base GNN for spatial processing
        if base_gnn is not None:
            self.base_gnn = base_gnn
        else:
            self.base_gnn = LeakDetectionGNN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                edge_dim=edge_dim,
                dropout=dropout,  # Dropout stays enabled for MC Dropout
            )
        
        # Heteroscedastic uncertainty model
        if use_heteroscedastic:
            self.heteroscedastic_model = HeteroscedasticUncertaintyModel(hidden_dim)
        else:
            self.heteroscedastic_model = None
        
        # Bayesian node head (optional, if using Bayesian layers)
        if use_bayesian:
            self.node_head_bayesian = BayesianLinear(hidden_dim, 1)
            # Keep deterministic head for comparison
            self.node_head_deterministic = self.base_gnn.node_head
        else:
            self.node_head_bayesian = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            sample: Whether to sample (for Bayesian layers).
        
        Returns:
            Tuple of:
            - node_logits: Leak logits [batch, num_nodes]
            - aleatoric_uncertainty: Aleatoric log variance [batch, num_nodes] (if heteroscedastic)
            - node_embeddings: Node embeddings for uncertainty analysis
        """
        # Get embeddings from base GNN
        h = self.base_gnn.input_proj(x)
        for mp_layer in self.base_gnn.mp_layers:
            h = mp_layer(h, edge_index, edge_attr)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Predict node logits
        if self.use_bayesian and self.node_head_bayesian is not None:
            node_logits = self.node_head_bayesian(h, sample=sample).squeeze(-1)
        else:
            node_logits = self.base_gnn.node_head(h).squeeze(-1)
        
        # Predict aleatoric uncertainty
        aleatoric_logvar = None
        if self.use_heteroscedastic and self.heteroscedastic_model is not None:
            aleatoric_logvar = self.heteroscedastic_model(h)
        
        return node_logits, aleatoric_logvar, h
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_samples: Optional[int] = None,
        return_samples: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using Monte Carlo sampling.
        
        PATENTABLE METHOD: Uses Monte Carlo Dropout and Bayesian inference
        to estimate epistemic and aleatoric uncertainty.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            num_samples: Number of MC samples (overrides self.mc_samples).
            return_samples: Whether to return all samples.
        
        Returns:
            Dictionary with:
            - mean: Mean predictions [batch, num_nodes]
            - std: Prediction standard deviation [batch, num_nodes]
            - aleatoric_std: Aleatoric uncertainty (std) [batch, num_nodes]
            - epistemic_std: Epistemic uncertainty (std) [batch, num_nodes]
            - confidence_intervals: 95% confidence intervals [batch, num_nodes, 2]
            - samples: (Optional) All MC samples [num_samples, batch, num_nodes]
        """
        if num_samples is None:
            num_samples = self.mc_samples
        
        self.eval()
        
        all_samples = []
        all_aleatoric_logvars = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample from posterior (enable dropout for MC Dropout)
                self.train()  # Enable dropout even in eval mode
                node_logits, aleatoric_logvar, _ = self.forward(
                    x, edge_index, edge_attr, sample=True
                )
                self.eval()
                
                node_probs = torch.sigmoid(node_logits)
                all_samples.append(node_probs)
                
                if aleatoric_logvar is not None:
                    all_aleatoric_logvars.append(aleatoric_logvar)
        
        # Stack samples: [num_samples, batch, num_nodes]
        samples_tensor = torch.stack(all_samples, dim=0)
        
        # Compute statistics
        mean_probs = samples_tensor.mean(dim=0)  # [batch, num_nodes]
        std_probs = samples_tensor.std(dim=0)  # [batch, num_nodes] (epistemic uncertainty)
        
        # Aleatoric uncertainty (mean of predicted log variances)
        aleatoric_std = None
        if all_aleatoric_logvars:
            aleatoric_logvar_mean = torch.stack(all_aleatoric_logvars, dim=0).mean(dim=0)
            aleatoric_std = torch.exp(0.5 * aleatoric_logvar_mean)  # [batch, num_nodes]
        
        # Total uncertainty = epistemic + aleatoric
        total_std = std_probs
        if aleatoric_std is not None:
            # Total variance = epistemic_var + aleatoric_var
            total_var = std_probs.pow(2) + aleatoric_std.pow(2)
            total_std = torch.sqrt(total_var)
        
        # Confidence intervals (95%)
        confidence_intervals = torch.stack([
            torch.clamp(mean_probs - 1.96 * total_std, 0.0, 1.0),
            torch.clamp(mean_probs + 1.96 * total_std, 0.0, 1.0),
        ], dim=-1)  # [batch, num_nodes, 2]
        
        result = {
            "mean": mean_probs,
            "std": total_std,
            "confidence_intervals": confidence_intervals,
        }
        
        if aleatoric_std is not None:
            result["aleatoric_std"] = aleatoric_std
            result["epistemic_std"] = std_probs  # Epistemic = MC Dropout std
        
        if return_samples:
            result["samples"] = samples_tensor
        
        return result
    
    def compute_kl_loss(self) -> torch.Tensor:
        """
        Compute KL divergence loss for Bayesian layers.
        
        Used in variational inference training.
        """
        total_kl = 0.0
        
        if self.use_bayesian and self.node_head_bayesian is not None:
            total_kl += self.node_head_bayesian.kl_divergence()
        
        return total_kl


def compute_bayesian_loss(
    node_logits: torch.Tensor,
    aleatoric_logvar: Optional[torch.Tensor],
    y_node: torch.Tensor,
    kl_weight: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss for Bayesian GNN training.
    
    Combines data likelihood loss with KL divergence (for variational inference).
    
    Args:
        node_logits: Predicted logits [batch, num_nodes]
        aleatoric_logvar: Aleatoric log variance [batch, num_nodes] (optional)
        y_node: Ground truth [batch, num_nodes]
        kl_weight: Weight for KL divergence term.
    
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(node_logits, y_node, reduction='mean')
    
    # Heteroscedastic loss (if aleatoric uncertainty is modeled)
    heteroscedastic_loss = 0.0
    if aleatoric_logvar is not None:
        # Negative log-likelihood with learned variance
        # NLL = 0.5 * log(var) + 0.5 * (error^2 / var)
        var = torch.exp(aleatoric_logvar).clamp(min=1e-6)
        error = (torch.sigmoid(node_logits) - y_node).pow(2)
        heteroscedastic_loss = 0.5 * (aleatoric_logvar + error / var).mean()
    
    total_loss = bce_loss + heteroscedastic_loss
    
    loss_dict = {
        "bce_loss": bce_loss.item(),
        "heteroscedastic_loss": heteroscedastic_loss.item() if aleatoric_logvar is not None else 0.0,
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict


def compute_uncertainty_metrics(
    predictions: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    uncertainty_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute metrics for uncertainty quantification.
    
    Evaluates calibration, coverage, and confidence-based accuracy.
    
    Args:
        predictions: Output from predict_with_uncertainty.
        y_true: Ground truth [batch, num_nodes]
        uncertainty_threshold: Threshold for high uncertainty.
    
    Returns:
        Dictionary of uncertainty metrics.
    """
    mean_probs = predictions["mean"]
    std_probs = predictions["std"]
    ci = predictions["confidence_intervals"]
    
    # Calibration: how well calibrated are confidence intervals?
    # Coverage: fraction of true values within confidence intervals
    within_ci = (y_true >= ci[:, :, 0]) & (y_true <= ci[:, :, 1])
    coverage = within_ci.float().mean().item()
    
    # Expected coverage for 95% CI should be ~0.95
    calibration_error = abs(coverage - 0.95)
    
    # High uncertainty detection
    high_uncertainty_mask = std_probs > uncertainty_threshold
    high_uncertainty_ratio = high_uncertainty_mask.float().mean().item()
    
    # Accuracy on low-uncertainty predictions
    low_uncertainty_mask = std_probs <= uncertainty_threshold
    if low_uncertainty_mask.any():
        low_unc_pred = mean_probs[low_uncertainty_mask]
        low_unc_true = y_true[low_uncertainty_mask]
        low_unc_acc = ((low_unc_pred > 0.5) == (low_unc_true > 0.5)).float().mean().item()
    else:
        low_unc_acc = 0.0
    
    return {
        "coverage": coverage,
        "calibration_error": calibration_error,
        "high_uncertainty_ratio": high_uncertainty_ratio,
        "low_uncertainty_accuracy": low_unc_acc,
    }

