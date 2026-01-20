#!/usr/bin/env python3
"""
Data Generation Script for Water Leak Localization.

Generates train, validation, and test datasets with synthetic leaks.

Usage:
    python scripts/generate_data.py --config configs/default.yaml
    python scripts/generate_data.py --num-nodes 100 --train-samples 500
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.sim.dataset_gen import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic water leak datasets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes in graph (overrides config)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of training samples (overrides config)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Number of validation samples (overrides config)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Number of test samples (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.num_nodes is not None:
        config.graph.num_nodes = args.num_nodes
    if args.train_samples is not None:
        config.data.train_samples = args.train_samples
    if args.val_samples is not None:
        config.data.val_samples = args.val_samples
    if args.test_samples is not None:
        config.data.test_samples = args.test_samples
    if args.seed is not None:
        config.seed = args.seed
    if args.output_dir is not None:
        config.data.output_dir = args.output_dir
    
    # Set seed
    set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Water Leak Dataset Generation")
    print("=" * 60)
    print(f"\nGraph: {config.graph.num_nodes} nodes")
    print(f"Sensor ratio: {config.sensors.ratio}")
    print(f"Noise std: {config.sensors.noise_std}")
    print(f"Leak probability: {config.leak.probability}")
    print(f"Leak magnitude: [{config.leak.magnitude_min}, {config.leak.magnitude_max}]")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate train set
    print(f"Generating {config.data.train_samples} training samples...")
    generate_dataset(
        num_samples=config.data.train_samples,
        num_nodes=config.graph.num_nodes,
        edge_probability=config.graph.edge_probability,
        num_reservoirs=config.graph.reservoir_nodes,
        sensor_ratio=config.sensors.ratio,
        demand_mean=config.demand.mean,
        demand_std=config.demand.std,
        reservoir_head=config.demand.reservoir_head,
        leak_probability=config.leak.probability,
        leak_magnitude_min=config.leak.magnitude_min,
        leak_magnitude_max=config.leak.magnitude_max,
        noise_std=config.sensors.noise_std,
        min_conductance=config.graph.min_conductance,
        max_conductance=config.graph.max_conductance,
        seed=config.seed,
        output_path=str(output_dir / "train.pt"),
        show_progress=True,
    )
    print(f"  Saved to {output_dir / 'train.pt'}")
    
    # Generate validation set
    print(f"\nGenerating {config.data.val_samples} validation samples...")
    generate_dataset(
        num_samples=config.data.val_samples,
        num_nodes=config.graph.num_nodes,
        edge_probability=config.graph.edge_probability,
        num_reservoirs=config.graph.reservoir_nodes,
        sensor_ratio=config.sensors.ratio,
        demand_mean=config.demand.mean,
        demand_std=config.demand.std,
        reservoir_head=config.demand.reservoir_head,
        leak_probability=config.leak.probability,
        leak_magnitude_min=config.leak.magnitude_min,
        leak_magnitude_max=config.leak.magnitude_max,
        noise_std=config.sensors.noise_std,
        min_conductance=config.graph.min_conductance,
        max_conductance=config.graph.max_conductance,
        seed=config.seed + 1000,  # Different seed for val
        output_path=str(output_dir / "val.pt"),
        show_progress=True,
    )
    print(f"  Saved to {output_dir / 'val.pt'}")
    
    # Generate test set
    print(f"\nGenerating {config.data.test_samples} test samples...")
    generate_dataset(
        num_samples=config.data.test_samples,
        num_nodes=config.graph.num_nodes,
        edge_probability=config.graph.edge_probability,
        num_reservoirs=config.graph.reservoir_nodes,
        sensor_ratio=config.sensors.ratio,
        demand_mean=config.demand.mean,
        demand_std=config.demand.std,
        reservoir_head=config.demand.reservoir_head,
        leak_probability=config.leak.probability,
        leak_magnitude_min=config.leak.magnitude_min,
        leak_magnitude_max=config.leak.magnitude_max,
        noise_std=config.sensors.noise_std,
        min_conductance=config.graph.min_conductance,
        max_conductance=config.graph.max_conductance,
        seed=config.seed + 2000,  # Different seed for test
        output_path=str(output_dir / "test.pt"),
        show_progress=True,
    )
    print(f"  Saved to {output_dir / 'test.pt'}")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
