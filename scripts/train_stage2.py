#!/usr/bin/env python3
"""
Stage 2 Training/Calibration Script for Leak Magnitude Estimation.

For physics-based approach, this script validates the estimation accuracy.
For learned approach, it trains the Stage 2 refiner.

Usage:
    python scripts/train_stage2.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.sim.dataset_gen import LeakDataset
from src.sim.simulator import HydraulicSimulator
from src.models.gnn_stage1 import LeakDetectionGNN
from src.models.stage2_refine import PhysicsBasedRefiner
from src.utils.graph import create_random_graph, select_reservoir_nodes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/validate Stage 2 magnitude estimation"
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
        default="checkpoints",
        help="Output directory",
    )
    parser.add_argument(
        "--use-physics",
        action="store_true",
        default=True,
        help="Use physics-based magnitude estimation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    return parser.parse_args()


def validate_physics_refiner(
    dataset: LeakDataset,
    stage1_model: LeakDetectionGNN,
    simulator: HydraulicSimulator,
    sensor_nodes: list,
    top_k: int = 5,
    device: str = "cpu",
) -> dict:
    """
    Validate the physics-based magnitude estimation.
    
    Args:
        dataset: Test dataset.
        stage1_model: Trained Stage 1 model.
        simulator: Hydraulic simulator.
        sensor_nodes: List of sensor nodes.
        top_k: Number of top candidates.
        device: Computation device.
    
    Returns:
        Dictionary of validation metrics.
    """
    refiner = PhysicsBasedRefiner(simulator, sensor_nodes)
    
    stage1_model.eval()
    
    magnitude_errors = []
    localization_correct = 0
    refined_localization_correct = 0
    total_leak_samples = 0
    
    for i in tqdm(range(len(dataset)), desc="Validating Stage 2"):
        sample = dataset[i]
        
        # Skip no-leak samples
        if sample["has_leak"] < 0.5:
            continue
        
        total_leak_samples += 1
        true_node = sample["leak_node"].item()
        true_magnitude = sample["leak_magnitude"].item()
        
        # Get Stage 1 predictions
        with torch.no_grad():
            node_features = sample["node_features"].unsqueeze(0).to(device)
            edge_index = sample["edge_index"].to(device)
            edge_attr = sample["edge_attr"].to(device)
            
            node_probs, _ = stage1_model.predict(node_features, edge_index, edge_attr)
            node_probs = node_probs.squeeze(0).cpu().numpy()
        
        # Get top-K candidates
        top_k_nodes = np.argsort(node_probs)[-top_k:][::-1].tolist()
        
        # Check Stage 1 localization
        if top_k_nodes[0] == true_node:
            localization_correct += 1
        
        # Get observed pressures (sensor readings from node_features)
        sensor_mask = sample["sensor_mask"].numpy()
        node_features_np = sample["node_features"].numpy()
        observed_pressures = node_features_np[sensor_mask > 0.5, 0]
        
        # Get base demand from dataset
        demand = dataset.demands[i].numpy()
        
        # Physics-based refinement
        best_node, best_magnitude, _ = refiner.refine_top_k(
            top_k_nodes, observed_pressures, demand
        )
        
        # Check refined localization
        if best_node == true_node:
            refined_localization_correct += 1
        
        # Compute magnitude error (only if localization is correct)
        if best_node == true_node:
            magnitude_errors.append(abs(best_magnitude - true_magnitude))
    
    # Compute metrics
    metrics = {
        "stage1_top1_accuracy": localization_correct / total_leak_samples,
        "refined_top1_accuracy": refined_localization_correct / total_leak_samples,
        "magnitude_mae": np.mean(magnitude_errors) if magnitude_errors else 0.0,
        "magnitude_std": np.std(magnitude_errors) if magnitude_errors else 0.0,
        "total_samples": total_leak_samples,
        "correctly_localized": refined_localization_correct,
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.seed is not None:
        config.seed = args.seed
    
    set_seed(config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    data_dir = Path(args.data_dir)
    val_dataset = LeakDataset(str(data_dir / "val.pt"))
    
    print("=" * 60)
    print("Stage 2 Magnitude Estimation Validation")
    print("=" * 60)
    
    # Recreate graph and simulator
    G = val_dataset.get_graph()
    num_nodes = val_dataset.num_nodes
    reservoir_nodes = val_dataset.reservoir_nodes
    sensor_nodes = val_dataset.data["sensor_nodes"]
    conductances = val_dataset.data["conductances"].numpy()
    
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=config.demand.reservoir_head,
        conductances=conductances,
        seed=config.seed,
    )
    
    # Load Stage 1 model
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"\nLoading Stage 1 model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        in_dim = val_dataset.node_features.shape[-1]
        edge_dim = val_dataset.edge_attr.shape[-1]
        
        stage1_model = LeakDetectionGNN(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            edge_dim=edge_dim,
            dropout=config.model.dropout,
        ).to(device)
        
        stage1_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"\nWarning: Checkpoint {checkpoint_path} not found!")
        print("Creating untrained model for testing...")
        
        in_dim = val_dataset.node_features.shape[-1]
        edge_dim = val_dataset.edge_attr.shape[-1]
        
        stage1_model = LeakDetectionGNN(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            edge_dim=edge_dim,
            dropout=config.model.dropout,
        ).to(device)
    
    # Validate physics-based refiner
    print("\nValidating physics-based magnitude estimation...")
    metrics = validate_physics_refiner(
        val_dataset,
        stage1_model,
        simulator,
        sensor_nodes,
        top_k=config.stage2.top_k,
        device=device,
    )
    
    print("\n" + "-" * 40)
    print("Validation Results:")
    print("-" * 40)
    print(f"Stage 1 Top-1 Accuracy: {metrics['stage1_top1_accuracy']:.4f}")
    print(f"Refined Top-1 Accuracy: {metrics['refined_top1_accuracy']:.4f}")
    print(f"Magnitude MAE: {metrics['magnitude_mae']:.4f}")
    print(f"Magnitude Std: {metrics['magnitude_std']:.4f}")
    print(f"Total leak samples: {metrics['total_samples']}")
    print(f"Correctly localized: {metrics['correctly_localized']}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "metrics": metrics,
        "config": config.to_dict(),
        "use_physics": args.use_physics,
    }, output_dir / "stage2_results.pt")
    
    print(f"\nResults saved to {output_dir / 'stage2_results.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
