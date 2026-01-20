#!/usr/bin/env python3
"""
Evaluation Script for Water Leak Localization.

Evaluates all models (GNN, baselines) and generates metrics and plots.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --test-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import (
    compute_roc_auc,
    compute_roc_curve,
    compute_top_k_accuracy_batch,
    compute_mean_graph_distance_error,
    compute_magnitude_mae,
    format_metrics,
)
from src.utils.visualization import plot_roc_curve, plot_top_k_bar
from src.sim.dataset_gen import LeakDataset
from src.sim.simulator import HydraulicSimulator
from src.models.gnn_stage1 import LeakDetectionGNN
from src.models.stage2_refine import PhysicsBasedRefiner
from src.baselines.residual_baseline import ResidualRankingBaseline
from src.baselines.graph_distance_baseline import GraphDistanceBaseline
from src.baselines.mlp_baseline import MLPBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate water leak localization models"
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
        "--checkpoint",
        type=str,
        default="checkpoints/best_stage1.pt",
        help="Stage 1 model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only evaluate on test set (skip train/val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    return parser.parse_args()


def evaluate_gnn(
    model: LeakDetectionGNN,
    dataset: LeakDataset,
    simulator: HydraulicSimulator,
    sensor_nodes: list,
    top_k_values: list,
    device: str = "cpu",
) -> dict:
    """Evaluate the GNN model."""
    model.eval()
    
    all_node_probs = []
    all_has_leak_pred = []
    all_has_leak = []
    all_leak_nodes = []
    all_leak_magnitudes = []
    all_pred_magnitudes = []
    
    refiner = PhysicsBasedRefiner(simulator, sensor_nodes)
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating GNN"):
            sample = dataset[i]
            
            node_features = sample["node_features"].unsqueeze(0).to(device)
            edge_index = sample["edge_index"].to(device)
            edge_attr = sample["edge_attr"].to(device)
            
            node_probs, no_leak_prob = model.predict(node_features, edge_index, edge_attr)
            node_probs = node_probs.squeeze(0).cpu().numpy()
            has_leak_pred = 1 - no_leak_prob.item()
            
            all_node_probs.append(node_probs)
            all_has_leak_pred.append(has_leak_pred)
            all_has_leak.append(sample["has_leak"].item())
            all_leak_nodes.append(sample["leak_node"].item())
            all_leak_magnitudes.append(sample["leak_magnitude"].item())
            
            # Estimate magnitude for top candidate
            if sample["has_leak"] > 0.5:
                top_node = np.argmax(node_probs)
                sensor_mask = sample["sensor_mask"].numpy()
                observed_pressures = sample["node_features"].numpy()[sensor_mask > 0.5, 0]
                demand = dataset.demands[i].numpy()
                
                try:
                    _, pred_mag, _ = refiner.refine_top_k(
                        [top_node], observed_pressures, demand
                    )
                except:
                    pred_mag = 0.0
                all_pred_magnitudes.append(pred_mag)
            else:
                all_pred_magnitudes.append(0.0)
    
    # Convert to arrays
    node_probs = np.array(all_node_probs)
    has_leak = np.array(all_has_leak)
    has_leak_pred = np.array(all_has_leak_pred)
    leak_nodes = np.array(all_leak_nodes)
    leak_magnitudes = np.array(all_leak_magnitudes)
    pred_magnitudes = np.array(all_pred_magnitudes)
    
    # Compute metrics
    G = dataset.get_graph()
    
    metrics = {
        "roc_auc": compute_roc_auc(has_leak, has_leak_pred),
    }
    
    for k in top_k_values:
        metrics[f"top_{k}_accuracy"] = compute_top_k_accuracy_batch(
            leak_nodes, node_probs, k
        )
    
    metrics["mean_graph_distance"] = compute_mean_graph_distance_error(
        G, leak_nodes, node_probs
    )
    
    leak_mask = has_leak > 0.5
    metrics["magnitude_mae"] = compute_magnitude_mae(
        leak_magnitudes, pred_magnitudes, leak_mask
    )
    
    # Store arrays for ROC curve
    metrics["_has_leak"] = has_leak
    metrics["_has_leak_pred"] = has_leak_pred
    
    return metrics


def evaluate_baseline(
    baseline,
    dataset: LeakDataset,
    baseline_name: str,
    top_k_values: list,
) -> dict:
    """Evaluate a baseline method."""
    
    all_node_probs = []
    all_has_leak = []
    all_leak_nodes = []
    
    for i in tqdm(range(len(dataset)), desc=f"Evaluating {baseline_name}"):
        sample = dataset[i]
        
        sensor_mask = sample["sensor_mask"].numpy()
        observed_pressures = sample["node_features"].numpy()[sensor_mask > 0.5, 0]
        
        # Get baseline pressures (no leak scenario approximation)
        # Use mean pressure as baseline
        baseline_pressures = np.full_like(observed_pressures, observed_pressures.mean())
        
        demand = dataset.demands[i].numpy()
        
        if hasattr(baseline, 'compute_sensitivity'):
            baseline.compute_sensitivity(demand)
        
        if hasattr(baseline, 'compute_scores') and hasattr(baseline, 'baseline_pressures'):
            scores = baseline.compute_scores(observed_pressures)
        else:
            # Graph distance baseline
            _, probs = baseline.predict(observed_pressures, baseline_pressures)
            scores = probs
        
        # Normalize scores to probabilities
        if isinstance(scores, np.ndarray):
            scores_valid = scores.copy()
            scores_valid[scores_valid == -np.inf] = -1e10
            exp_scores = np.exp(scores_valid - np.max(scores_valid))
            probs = exp_scores / (exp_scores.sum() + 1e-10)
        else:
            probs = scores
        
        all_node_probs.append(probs)
        all_has_leak.append(sample["has_leak"].item())
        all_leak_nodes.append(sample["leak_node"].item())
    
    # Convert to arrays
    node_probs = np.array(all_node_probs)
    has_leak = np.array(all_has_leak)
    leak_nodes = np.array(all_leak_nodes)
    
    # Compute metrics
    G = dataset.get_graph()
    
    metrics = {}
    for k in top_k_values:
        metrics[f"top_{k}_accuracy"] = compute_top_k_accuracy_batch(
            leak_nodes, node_probs, k
        )
    
    metrics["mean_graph_distance"] = compute_mean_graph_distance_error(
        G, leak_nodes, node_probs
    )
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.seed is not None:
        config.seed = args.seed
    
    set_seed(config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test dataset
    data_dir = Path(args.data_dir)
    test_dataset = LeakDataset(str(data_dir / "test.pt"))
    
    print("=" * 60)
    print("Water Leak Localization Evaluation")
    print("=" * 60)
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Device: {device}")
    
    # Setup simulator
    G = test_dataset.get_graph()
    num_nodes = test_dataset.num_nodes
    reservoir_nodes = test_dataset.reservoir_nodes
    sensor_nodes = test_dataset.data["sensor_nodes"]
    conductances = test_dataset.data["conductances"].numpy()
    
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=config.demand.reservoir_head,
        conductances=conductances,
        seed=config.seed,
    )
    
    top_k_values = config.eval.top_k_values
    
    # Evaluate GNN
    print("\n" + "-" * 40)
    print("Evaluating GNN Model")
    print("-" * 40)
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        in_dim = test_dataset.node_features.shape[-1]
        edge_dim = test_dataset.edge_attr.shape[-1]
        
        gnn_model = LeakDetectionGNN(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            edge_dim=edge_dim,
            dropout=0.0,  # No dropout during evaluation
        ).to(device)
        
        gnn_model.load_state_dict(checkpoint["model_state_dict"])
        gnn_metrics = evaluate_gnn(
            gnn_model, test_dataset, simulator, sensor_nodes, top_k_values, device
        )
        
        print("\nGNN Results:")
        print(f"  ROC-AUC: {gnn_metrics['roc_auc']:.4f}")
        for k in top_k_values:
            print(f"  Top-{k}: {gnn_metrics[f'top_{k}_accuracy']:.4f}")
        print(f"  Mean Graph Distance: {gnn_metrics['mean_graph_distance']:.2f}")
        print(f"  Magnitude MAE: {gnn_metrics['magnitude_mae']:.4f}")
        
        # Generate ROC curve
        fpr, tpr, _ = compute_roc_curve(
            gnn_metrics["_has_leak"],
            gnn_metrics["_has_leak_pred"]
        )
        plot_roc_curve(fpr, tpr, gnn_metrics["roc_auc"], 
                       save_path=str(output_dir / "roc_curve.png"))
        print(f"\n  ROC curve saved to {output_dir / 'roc_curve.png'}")
    else:
        print(f"  Warning: Checkpoint {checkpoint_path} not found, skipping GNN evaluation")
        gnn_metrics = None
    
    # Evaluate Residual Baseline
    print("\n" + "-" * 40)
    print("Evaluating Residual Baseline")
    print("-" * 40)
    
    residual_baseline = ResidualRankingBaseline(simulator, sensor_nodes)
    residual_metrics = evaluate_baseline(
        residual_baseline, test_dataset, "Residual", top_k_values
    )
    
    print("\nResidual Baseline Results:")
    for k in top_k_values:
        print(f"  Top-{k}: {residual_metrics[f'top_{k}_accuracy']:.4f}")
    print(f"  Mean Graph Distance: {residual_metrics['mean_graph_distance']:.2f}")
    
    # Evaluate Graph Distance Baseline
    print("\n" + "-" * 40)
    print("Evaluating Graph Distance Baseline")
    print("-" * 40)
    
    graph_baseline = GraphDistanceBaseline(G, sensor_nodes, reservoir_nodes)
    graph_metrics = evaluate_baseline(
        graph_baseline, test_dataset, "Graph Distance", top_k_values
    )
    
    print("\nGraph Distance Baseline Results:")
    for k in top_k_values:
        print(f"  Top-{k}: {graph_metrics[f'top_{k}_accuracy']:.4f}")
    print(f"  Mean Graph Distance: {graph_metrics['mean_graph_distance']:.2f}")
    
    # Generate comparison bar chart
    print("\n" + "-" * 40)
    print("Generating Comparison Plot")
    print("-" * 40)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(top_k_values))
    width = 0.25
    
    if gnn_metrics:
        gnn_accs = [gnn_metrics[f"top_{k}_accuracy"] for k in top_k_values]
        ax.bar(x - width, gnn_accs, width, label="GNN", color="steelblue")
    
    residual_accs = [residual_metrics[f"top_{k}_accuracy"] for k in top_k_values]
    ax.bar(x, residual_accs, width, label="Residual", color="darkorange")
    
    graph_accs = [graph_metrics[f"top_{k}_accuracy"] for k in top_k_values]
    ax.bar(x + width, graph_accs, width, label="Graph Distance", color="forestgreen")
    
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Accuracy")
    ax.set_title("Leak Localization Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in top_k_values])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    print(f"Comparison plot saved to {output_dir / 'comparison.png'}")
    
    # Save all results
    results = {
        "gnn": gnn_metrics if gnn_metrics else {},
        "residual": residual_metrics,
        "graph_distance": graph_metrics,
        "config": config.to_dict(),
    }
    torch.save(results, output_dir / "evaluation_results.pt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if gnn_metrics:
        print("\n[GNN Model]")
        print(f"  ROC-AUC: {gnn_metrics['roc_auc']:.4f}")
        print(f"  Top-1 Accuracy: {gnn_metrics['top_1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {gnn_metrics['top_5_accuracy']:.4f}")
        print(f"  Mean Graph Distance: {gnn_metrics['mean_graph_distance']:.2f}")
        print(f"  Magnitude MAE: {gnn_metrics['magnitude_mae']:.4f}")
    
    print("\n[Residual Baseline]")
    print(f"  Top-1 Accuracy: {residual_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {residual_metrics['top_5_accuracy']:.4f}")
    
    print("\n[Graph Distance Baseline]")
    print(f"  Top-1 Accuracy: {graph_metrics['top_1_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {graph_metrics['top_5_accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Results saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
