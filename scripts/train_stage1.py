#!/usr/bin/env python3
"""
Stage 1 Training Script for Water Leak Detection GNN.

Trains the GNN for leak detection and localization.

Usage:
    python scripts/train_stage1.py --config configs/default.yaml
    python scripts/train_stage1.py --epochs 100 --lr 0.001
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.sim.dataset_gen import LeakDataset, create_dataloader
from src.models.gnn_stage1 import LeakDetectionGNN, compute_stage1_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Stage 1 GNN for leak detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cpu/cuda)",
    )
    return parser.parse_args()


def train_epoch(
    model: LeakDetectionGNN,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str,
    focal_gamma: float,
    no_leak_weight: float,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_node_loss = 0.0
    total_no_leak_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        node_features = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        y_node = batch["y_node"].to(device)
        has_leak = batch["has_leak"].to(device)
        
        optimizer.zero_grad()
        
        node_logits, no_leak_logit = model(node_features, edge_index, edge_attr)
        
        loss, loss_dict = compute_stage1_loss(
            node_logits, no_leak_logit, y_node, has_leak,
            focal_gamma=focal_gamma, no_leak_weight=no_leak_weight
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_node_loss += loss_dict["node_loss"]
        total_no_leak_loss += loss_dict["no_leak_loss"]
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "node_loss": total_node_loss / num_batches,
        "no_leak_loss": total_no_leak_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: LeakDetectionGNN,
    dataloader,
    device: str,
) -> dict:
    """Evaluate the model."""
    model.eval()
    
    all_node_probs = []
    all_no_leak_probs = []
    all_y_node = []
    all_has_leak = []
    all_leak_nodes = []
    
    for batch in dataloader:
        node_features = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        
        node_probs, no_leak_prob = model.predict(node_features, edge_index, edge_attr)
        
        all_node_probs.append(node_probs.cpu())
        all_no_leak_probs.append(no_leak_prob.cpu())
        all_y_node.append(batch["y_node"])
        all_has_leak.append(batch["has_leak"])
        all_leak_nodes.append(batch["leak_node"])
    
    node_probs = torch.cat(all_node_probs, dim=0).numpy()
    no_leak_probs = torch.cat(all_no_leak_probs, dim=0).numpy()
    y_node = torch.cat(all_y_node, dim=0).numpy()
    has_leak = torch.cat(all_has_leak, dim=0).numpy()
    leak_nodes = torch.cat(all_leak_nodes, dim=0).numpy()
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score
    
    # Detection AUC
    has_leak_pred = 1 - no_leak_probs.flatten()
    if len(set(has_leak)) > 1:
        detection_auc = roc_auc_score(has_leak, has_leak_pred)
    else:
        detection_auc = 0.5
    
    # Localization accuracy (only for samples with leaks)
    leak_mask = has_leak > 0.5
    correct_top1 = 0
    correct_top5 = 0
    total = leak_mask.sum()
    
    for i in range(len(leak_nodes)):
        if not leak_mask[i]:
            continue
        
        true_node = leak_nodes[i]
        predicted_top = node_probs[i].argsort()[-5:][::-1]
        
        if predicted_top[-1] == true_node:
            correct_top1 += 1
        if true_node in predicted_top:
            correct_top5 += 1
    
    return {
        "detection_auc": detection_auc,
        "top1_accuracy": correct_top1 / total if total > 0 else 0.0,
        "top5_accuracy": correct_top5 / total if total > 0 else 0.0,
    }


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed
    
    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(args.data_dir)
    train_loader = create_dataloader(
        str(data_dir / "train.pt"),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = create_dataloader(
        str(data_dir / "val.pt"),
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    
    # Get input dimensions from data
    train_dataset = LeakDataset(str(data_dir / "train.pt"))
    in_dim = train_dataset.node_features.shape[-1]
    edge_dim = train_dataset.edge_attr.shape[-1]
    
    print("=" * 60)
    print("Stage 1 GNN Training")
    print("=" * 60)
    print(f"\nDevice: {device}")
    print(f"Input dim: {in_dim}, Edge dim: {edge_dim}")
    print(f"Hidden dim: {config.model.hidden_dim}")
    print(f"Num layers: {config.model.num_layers}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print()
    
    # Create model
    model = LeakDetectionGNN(
        in_dim=in_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        edge_dim=edge_dim,
        dropout=config.model.dropout,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(config.training.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            focal_gamma=config.training.focal_gamma,
            no_leak_weight=config.training.no_leak_weight,
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Step scheduler
        scheduler.step(val_metrics["detection_auc"])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.training.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val AUC: {val_metrics['detection_auc']:.4f} | "
              f"Top-1: {val_metrics['top1_accuracy']:.4f} | "
              f"Top-5: {val_metrics['top5_accuracy']:.4f}")
        
        # Save best model
        if val_metrics["detection_auc"] > best_val_auc:
            best_val_auc = val_metrics["detection_auc"]
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "config": config.to_dict(),
            }, output_dir / "best_stage1.pt")
            print(f"  -> Saved best model (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
    }, output_dir / "final_stage1.pt")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best Val AUC: {best_val_auc:.4f}")
    print(f"Checkpoints saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
