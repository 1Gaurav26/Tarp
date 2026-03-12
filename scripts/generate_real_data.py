#!/usr/bin/env python3
"""
Data Generation from Real Water Networks.

Generates train, validation, and test datasets using a real EPANET .inp
network topology (e.g., BattLeDIM L-Town).

Usage:
    python scripts/generate_real_data.py --inp-file data/networks/L-TOWN.inp
    python scripts/generate_real_data.py --inp-file data/networks/L-TOWN.inp --train-samples 5000
    python scripts/generate_real_data.py --config configs/real_ltown.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.sim.dataset_gen_real import generate_real_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate leak datasets from real water network .inp files"
    )
    parser.add_argument(
        "--inp-file",
        type=str,
        default="data/networks/L-TOWN.inp",
        help="Path to EPANET .inp file (default: L-Town)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/real",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=800,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=800,
        help="Number of test samples",
    )
    parser.add_argument(
        "--sensor-ratio",
        type=float,
        default=0.25,
        help="Fraction of nodes used as sensors",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Sensor noise standard deviation",
    )
    parser.add_argument(
        "--leak-prob",
        type=float,
        default=0.8,
        help="Probability of leak in each sample",
    )
    parser.add_argument(
        "--leak-min",
        type=float,
        default=5.0,
        help="Minimum leak magnitude",
    )
    parser.add_argument(
        "--leak-max",
        type=float,
        default=50.0,
        help="Maximum leak magnitude",
    )
    parser.add_argument(
        "--reservoir-head",
        type=float,
        default=50.0,
        help="Fixed head at reservoirs (meters)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real Network Dataset Generation")
    print("=" * 60)
    print(f"\nNetwork file: {args.inp_file}")
    print(f"Sensor ratio: {args.sensor_ratio}")
    print(f"Noise std: {args.noise_std}")
    print(f"Leak probability: {args.leak_prob}")
    print(f"Leak magnitude: [{args.leak_min}, {args.leak_max}]")
    print(f"Output directory: {output_dir}")
    print()

    # Common kwargs shared across train/val/test
    common = dict(
        inp_path=args.inp_file,
        sensor_ratio=args.sensor_ratio,
        demand_mean=1.0,
        demand_std=0.5,
        reservoir_head=args.reservoir_head,
        leak_probability=args.leak_prob,
        leak_magnitude_min=args.leak_min,
        leak_magnitude_max=args.leak_max,
        noise_std=args.noise_std,
        show_progress=True,
    )

    # ── Train set ──────────────────────────────────────────────────
    print(f"Generating {args.train_samples} training samples...")
    generate_real_dataset(
        num_samples=args.train_samples,
        seed=args.seed,
        output_path=str(output_dir / "train.pt"),
        **common,
    )
    print(f"  Saved to {output_dir / 'train.pt'}")

    # ── Validation set ─────────────────────────────────────────────
    print(f"\nGenerating {args.val_samples} validation samples...")
    generate_real_dataset(
        num_samples=args.val_samples,
        seed=args.seed + 1000,
        output_path=str(output_dir / "val.pt"),
        **common,
    )
    print(f"  Saved to {output_dir / 'val.pt'}")

    # ── Test set ───────────────────────────────────────────────────
    print(f"\nGenerating {args.test_samples} test samples...")
    generate_real_dataset(
        num_samples=args.test_samples,
        seed=args.seed + 2000,
        output_path=str(output_dir / "test.pt"),
        **common,
    )
    print(f"  Saved to {output_dir / 'test.pt'}")

    print("\n" + "=" * 60)
    print("Real network dataset generation complete!")
    print("=" * 60)
    print(f"\nTo train on this data, update your config or run:")
    print(f"  python scripts/train_stage1.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
