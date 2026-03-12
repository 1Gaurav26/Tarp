#!/usr/bin/env python3
"""
Stage 1 Training Script for Water Leak Detection GNN (Production).

Features:
- Cosine annealing LR with linear warmup
- Mixed-precision training (AMP) on GPU
- Gradient accumulation for larger effective batch
- Comprehensive metric tracking (AUC, Top-1, Top-5)
- Structured logging with epoch timing

Usage:
    python scripts/train_stage1.py --config configs/default.yaml
    python scripts/train_stage1.py --epochs 100 --lr 0.0005
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.sim.dataset_gen import LeakDataset, create_dataloader
from src.models.gnn_stage1 import LeakDetectionGNN, compute_stage1_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 GNN (Production)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Cosine annealing LR scheduler with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model, dataloader, optimizer, scheduler, device,
    focal_gamma, no_leak_weight,
    scaler=None, grad_accum_steps=1,
):
    """Train one epoch with gradient accumulation and optional AMP."""
    model.train()
    total_loss = 0.0
    total_node_loss = 0.0
    total_no_leak_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        node_features = batch["node_features"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        y_node = batch["y_node"].to(device)
        has_leak = batch["has_leak"].to(device)

        # Mixed precision forward
        use_amp = scaler is not None and device != "cpu"
        with torch.amp.autocast("cuda", enabled=use_amp):
            node_logits, no_leak_logit = model(node_features, edge_index, edge_attr)
            loss, loss_dict = compute_stage1_loss(
                node_logits, no_leak_logit, y_node, has_leak,
                focal_gamma=focal_gamma, no_leak_weight=no_leak_weight,
                edge_index=edge_index,
            )
            loss = loss / grad_accum_steps

        # Backward
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum_steps
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_dict["total_loss"]
        total_node_loss += loss_dict["node_loss"]
        total_no_leak_loss += loss_dict["no_leak_loss"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "node_loss": total_node_loss / num_batches,
        "no_leak_loss": total_no_leak_loss / num_batches,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate with comprehensive metrics."""
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

    from sklearn.metrics import roc_auc_score

    # Detection AUC
    has_leak_pred = 1 - no_leak_probs.flatten()
    if len(set(has_leak.flatten())) > 1:
        detection_auc = roc_auc_score(has_leak.flatten(), has_leak_pred)
    else:
        detection_auc = 0.5

    # Localization accuracy
    leak_mask = has_leak.flatten() > 0.5
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = leak_mask.sum()

    for i in range(len(leak_nodes)):
        if not leak_mask[i]:
            continue
        true_node = leak_nodes[i]
        ranked = node_probs[i].argsort()[::-1]

        if ranked[0] == true_node:
            correct_top1 += 1
        if true_node in ranked[:3]:
            correct_top3 += 1
        if true_node in ranked[:5]:
            correct_top5 += 1

    return {
        "detection_auc": detection_auc,
        "top1_accuracy": correct_top1 / total if total > 0 else 0.0,
        "top3_accuracy": correct_top3 / total if total > 0 else 0.0,
        "top5_accuracy": correct_top5 / total if total > 0 else 0.0,
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.seed is not None:
        config.seed = args.seed

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.training.use_amp and device == "cuda" and not args.no_amp

    set_seed(config.seed)

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

    # Infer dimensions from data
    train_dataset = LeakDataset(str(data_dir / "train.pt"))
    in_dim = train_dataset.node_features.shape[-1]
    edge_dim = train_dataset.edge_attr.shape[-1]
    num_train = len(train_dataset)

    print("=" * 70)
    print("  Stage 1 GNN Training (Production)")
    print("=" * 70)
    print(f"  Device:           {device} {'(AMP)' if use_amp else ''}")
    print(f"  Input dim:        {in_dim}")
    print(f"  Hidden dim:       {config.model.hidden_dim}")
    print(f"  Layers:           {config.model.num_layers}")
    print(f"  Attention heads:  {config.model.num_heads}")
    print(f"  Learning rate:    {config.training.learning_rate}")
    print(f"  Epochs:           {config.training.epochs}")
    print(f"  Batch size:       {config.training.batch_size} (eff: {config.training.batch_size * config.training.grad_accum_steps})")
    print(f"  Train samples:    {num_train}")
    print(f"  Scheduler:        {config.training.scheduler}")
    print(f"  Warmup:           {config.training.warmup_epochs} epochs")
    print()

    # Create model
    model = LeakDetectionGNN(
        in_dim=in_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        edge_dim=edge_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params:     {num_params:,}")
    print("=" * 70)
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / config.training.grad_accum_steps)
    total_steps = steps_per_epoch * config.training.epochs
    warmup_steps = steps_per_epoch * config.training.warmup_epochs

    if config.training.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        # Fallback: constant LR with warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=1.0)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training loop
    best_val_score = 0.0  # Combined score: AUC + Top-5
    patience_counter = 0
    history = []

    for epoch in range(config.training.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            focal_gamma=config.training.focal_gamma,
            no_leak_weight=config.training.no_leak_weight,
            scaler=scaler,
            grad_accum_steps=config.training.grad_accum_steps,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Combined validation score for model selection
        val_score = 0.4 * val_metrics["detection_auc"] + 0.6 * val_metrics["top5_accuracy"]

        # Logging
        print(
            f"Epoch {epoch+1:3d}/{config.training.epochs} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"| AUC: {val_metrics['detection_auc']:.4f} "
            f"| Top-1: {val_metrics['top1_accuracy']:.4f} "
            f"| Top-3: {val_metrics['top3_accuracy']:.4f} "
            f"| Top-5: {val_metrics['top5_accuracy']:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| {elapsed:.1f}s"
        )

        history.append({**train_metrics, **val_metrics, "lr": current_lr, "epoch": epoch + 1})

        # Save best model
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "val_score": val_score,
                "config": config.to_dict(),
            }, output_dir / "best_stage1.pt")
            print(f"  -> Saved best model (score: {val_score:.4f})")
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
        "history": history,
    }, output_dir / "final_stage1.pt")

    print("\n" + "=" * 70)
    print(f"  Training complete!")
    print(f"  Best validation score:  {best_val_score:.4f}")
    if history:
        best_epoch = max(history, key=lambda h: 0.4 * h["detection_auc"] + 0.6 * h["top5_accuracy"])
        print(f"  Best epoch:             {best_epoch['epoch']}")
        print(f"    AUC:   {best_epoch['detection_auc']:.4f}")
        print(f"    Top-1: {best_epoch['top1_accuracy']:.4f}")
        print(f"    Top-3: {best_epoch['top3_accuracy']:.4f}")
        print(f"    Top-5: {best_epoch['top5_accuracy']:.4f}")
    print(f"  Checkpoints saved to:   {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
