# 💧 Water Leak Localization in Pipe Networks

### Graph ML + Synthetic Leaks | Production-Ready Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade machine learning pipeline for detecting and localizing water leaks in pipe networks using **Edge-Gated Multi-Head Attention GNNs** and physics-based estimation. Supports both **synthetic** and **real water network** topologies (BattLeDIM L-Town, EPANET `.inp` files).

![Demo Preview](results/comparison.png)

---

## 🎯 Overview

This project implements a **two-stage** approach for water leak localization:

```
Sensor Readings → [Stage 1: GNN Localization] → Top-K Leak Candidates
                                                        ↓
     Leak Location + Magnitude ← [Stage 2: Physics Solver] ←─┘
```

### Stage 1: Edge-Gated Attention GNN
- Multi-head attention with learnable edge gating by pipe conductance
- **7-dimensional enriched node features** (pressure, topology, graph structure)
- Per-node leak probability heatmap + global no-leak detection score
- Focal loss with **neighborhood label smoothing** for extreme class imbalance (1:200)

### Stage 2: Physics-Based Magnitude Estimation
- Closed-form least-squares solver using hydraulic model
- Sensitivity matrix from weighted Laplacian
- Minimizes sensor residual error to estimate leak magnitude

---

## 🏗️ Architecture

### GNN Model (`src/models/gnn_stage1.py`)

```
Input Features [B, N, 7]
        ↓
┌─────────────────────────┐
│   Input Projection      │  Linear → BatchNorm1d → GELU → Dropout
│   [B, N, 7] → [B, N, H]│
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Edge-Gated Attention   │  × 4 layers
│  Layer (Multi-Head)     │
│                         │
│  • Q/K/V projections    │  [N, heads, head_dim]
│  • Edge gate: σ(MLP(e)) │  Pipe conductance → per-head gate
│  • Attention: MLP(q,k,e)│  Learned score + edge gating
│  • Scatter softmax      │  Per-destination normalization
│  • Skip + LayerNorm     │  Residual connection
└─────────────────────────┘
        ↓
   ┌────┴────┐
   ↓         ↓
┌──────┐  ┌──────────────┐
│ Node │  │ Attention    │
│ Head │  │ Pooling      │
│ MLP  │  │ → Global MLP │
└──────┘  └──────────────┘
   ↓              ↓
Per-Node      No-Leak
Leak Probs    Score
```

### 7-Dim Node Features (`src/sim/dataset_gen.py`)

| Dim | Feature | Source | Purpose |
|-----|---------|--------|---------|
| 0 | Normalized pressure | Sensor readings | Primary signal |
| 1 | Sensor mask | Binary (0/1) | Identifies measured nodes |
| 2 | Node degree (norm.) | Graph topology | Structural context |
| 3 | Reservoir distance (norm.) | BFS distance | Proximity to sources |
| 4 | Is reservoir | Binary flag | Boundary condition |
| 5 | Mean neighbor conductance | Edge weights | Local pipe capacity |
| 6 | Pressure deviation | Sensor − global mean | Anomaly detection |

### Training Enhancements (`scripts/train_stage1.py`)

| Feature | Details |
|---------|---------|
| **Optimizer** | AdamW with weight decay 1e-4 |
| **LR Schedule** | Cosine annealing with linear warmup (5 epochs) |
| **Mixed Precision** | `torch.amp` (automatic on CUDA) |
| **Gradient Accumulation** | Effective batch = `batch_size × grad_accum_steps` |
| **Loss Function** | Focal Loss (α=0.9, γ=2.0) + Neighborhood Label Smoothing |
| **Model Selection** | Combined score: 0.4 × AUC + 0.6 × Top-5 accuracy |
| **Early Stopping** | Patience = 15 epochs |

### Loss Function Details

**Problem**: With 200 nodes and 1 leak, the class ratio is **1:199**. Standard BCE completely ignores the leak node.

**Solution**: Two techniques combined:

1. **Focal Loss with α=0.9** — Upweights the positive (leak) class by 9× relative to negative nodes
2. **Neighborhood Label Smoothing** — 1-hop neighbors of the leak node get partial labels (0.3), spreading the gradient signal from 1 node to ~15 nodes

---

## 📁 Project Structure

```
├── src/
│   ├── sim/
│   │   ├── simulator.py          # Linear hydraulic simulator (Laplacian solver)
│   │   ├── dataset_gen.py        # 7-dim enriched feature dataset generation
│   │   └── dataset_gen_real.py   # Real network dataset generation (EPANET .inp)
│   ├── models/
│   │   ├── gnn_stage1.py         # Edge-gated multi-head attention GNN
│   │   ├── stage2_refine.py      # Physics-based magnitude estimation
│   │   ├── gnn_multileak.py      # Multi-leak detection
│   │   ├── gnn_temporal.py       # Temporal-aware GNN
│   │   ├── gnn_bayesian.py       # Bayesian GNN with uncertainty
│   │   ├── sensor_placement.py   # Sensor placement optimization
│   │   ├── multimodal_fusion.py  # Multi-modal sensor fusion
│   │   ├── transfer_learning.py  # Cross-network transfer learning
│   │   └── pinn_hybrid.py        # Physics-Informed Neural Network
│   ├── baselines/
│   │   ├── residual_baseline.py  # Pressure residual ranking
│   │   ├── graph_distance_baseline.py
│   │   └── mlp_baseline.py
│   └── utils/
│       ├── config.py             # YAML config with dataclasses
│       ├── seed.py               # Reproducibility
│       ├── graph.py              # Graph generation + EPANET .inp loader
│       ├── metrics.py            # ROC-AUC, Top-K, graph distance
│       ├── visualization.py      # Plotly + Matplotlib plots
│       └── explainability.py     # SHAP, attention visualization
├── scripts/
│   ├── generate_data.py          # Synthetic dataset generation
│   ├── generate_real_data.py     # Real network dataset generation
│   ├── train_stage1.py           # GNN training (cosine LR, AMP, grad accum)
│   ├── train_stage2.py           # Stage 2 validation
│   └── evaluate.py               # Full evaluation + comparison plots
├── data/
│   └── networks/
│       └── L-TOWN.inp            # BattLeDIM L-Town network (785 nodes)
├── app/
│   └── streamlit_app.py          # Interactive demo
├── configs/
│   ├── default.yaml              # Synthetic network configuration
│   └── real_ltown.yaml           # L-Town real network configuration
├── docs/
│   ├── ADVANCED_FEATURES.md
│   ├── PATENT_STRATEGY.md
│   └── IMPLEMENTATION_COMPLETE.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/1Gaurav26/Tarp.git
cd Tarp

# Create virtual environment
python -m venv tarp
tarp\Scripts\activate         # Windows
# source tarp/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### 2. Generate Data

#### Option A: Synthetic Network (random graph)

```bash
# Production dataset (5000 train, 200 nodes)
python scripts/generate_data.py --train-samples 5000 --seed 42

# Quick test dataset
python scripts/generate_data.py --num-nodes 100 --train-samples 1000 --seed 42
```

#### Option B: Real Network (BattLeDIM L-Town)

```bash
# Production dataset from L-Town (785 nodes, 909 pipes)
python scripts/generate_real_data.py --train-samples 5000 --seed 42

# Quick test dataset
python scripts/generate_real_data.py --train-samples 500 --seed 42

# Use any EPANET .inp file
python scripts/generate_real_data.py --inp-file path/to/network.inp --train-samples 5000
```

### 3. Train

```bash
# Train on synthetic data (default)
python scripts/train_stage1.py --epochs 100

# Train on real L-Town data
python scripts/train_stage1.py --data-dir data/real --config configs/real_ltown.yaml

# Quick test
python scripts/train_stage1.py --epochs 20

# Validate Stage 2 magnitude estimation
python scripts/train_stage2.py
```

### 4. Evaluate

```bash
python scripts/evaluate.py
# Results + plots saved to results/
```

### 5. Demo App

```bash
streamlit run app/streamlit_app.py
```

---

## ⚙️ Configuration

All parameters in `configs/default.yaml`:

```yaml
# Graph
graph:
  num_nodes: 200
  edge_probability: 0.08
  reservoir_nodes: 2

# Sensors
sensors:
  ratio: 0.2               # 20% of nodes are sensors
  noise_std: 0.1

# Leak injection
leak:
  probability: 0.8
  magnitude_min: 0.5
  magnitude_max: 5.0

# Training (production-tuned)
training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 1.0e-4
  patience: 15
  grad_accum_steps: 2       # Effective batch = 128
  use_amp: true             # Mixed precision on GPU
  scheduler: "cosine"
  warmup_epochs: 5

# Model
model:
  in_dim: 7                 # Enriched features
  hidden_dim: 128
  num_layers: 4
  num_heads: 4
  dropout: 0.15

seed: 42
```

---

## 🔬 Hydraulic Simulator

The `HydraulicSimulator` solves the linear hydraulic system:

```
Pipe Flow:       Q_ij = g_e × (h_i − h_j)       (conductance × head difference)
Node Continuity: Σ_j Q_ij = d_i                  (flow balance at each node)
System Matrix:   L @ h = d                        (weighted Laplacian)
Leak Injection:  d_k' = d_k + q                  (additional demand at leak node k)
```

Reservoir nodes have fixed head `H₀ = 50m`. The sparse system is solved via `scipy.sparse.linalg.spsolve`.

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Leak detection accuracy (binary: leak vs no-leak) |
| **Top-1 Accuracy** | True leak node is the #1 prediction |
| **Top-3 Accuracy** | True leak node is within top 3 predictions |
| **Top-5 Accuracy** | True leak node is within top 5 predictions |
| **Mean Graph Distance** | Shortest path hops from prediction to true leak |
| **Magnitude MAE** | Mean absolute error of leak magnitude estimate |

---

## 📦 Data Format

Each `.pt` dataset file contains:

```python
{
    "node_features":  [N_samples, num_nodes, 7],   # 7-dim enriched features
    "edge_index":     [2, num_edges × 2],           # Bidirectional edges
    "edge_attr":      [num_edges × 2, 2],           # (conductance, length)
    "y_node":         [N_samples, num_nodes],        # One-hot leak labels
    "has_leak":       [N_samples],                   # Binary leak indicator
    "leak_nodes":     [N_samples],                   # Leak node indices
    "leak_magnitudes":[N_samples],                   # Leak magnitudes
    "sensor_mask":    [num_nodes],                    # Sensor node binary mask
    "full_pressures": [N_samples, num_nodes],        # True pressure field
    "demands":        [N_samples, num_nodes],         # Demand vectors
    "conductances":   [num_edges],                    # Pipe conductances
}
```

---

## 🔧 Python API

### Simulator

```python
from src.sim.simulator import HydraulicSimulator

simulator = HydraulicSimulator(G=graph, reservoir_nodes=[0, 1], reservoir_head=50.0)
demand = simulator.generate_demand(mean=1.0, std=0.5)
heads = simulator.solve(demand, leak_node=42, leak_magnitude=3.0)
readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std=0.1)
```

### GNN Model

```python
from src.models.gnn_stage1 import LeakDetectionGNN

model = LeakDetectionGNN(in_dim=7, hidden_dim=128, num_layers=4, num_heads=4)
node_probs, no_leak_prob = model.predict(x, edge_index, edge_attr)
```

### Physics Refiner

```python
from src.models.stage2_refine import PhysicsBasedRefiner

refiner = PhysicsBasedRefiner(simulator, sensor_nodes)
best_node, magnitude, estimates = refiner.refine_top_k(
    top_k_nodes=[24, 56, 78], observed_pressures=readings, base_demand=demand
)
```

---

## � Advanced Features

This repository includes 8 advanced modules for research and production:

| Module | File | Capability |
|--------|------|-----------|
| Multi-Leak Detection | `gnn_multileak.py` | Simultaneous multiple leak detection |
| Temporal GNN | `gnn_temporal.py` | Time-series leak evolution tracking |
| Bayesian GNN | `gnn_bayesian.py` | Uncertainty quantification (epistemic + aleatoric) |
| Sensor Placement | `sensor_placement.py` | Information-theoretic sensor optimization |
| Multi-Modal Fusion | `multimodal_fusion.py` | Pressure + flow + acoustic fusion |
| Transfer Learning | `transfer_learning.py` | Cross-network domain adaptation |
| Physics-Informed NN | `pinn_hybrid.py` | Hybrid physics + learned solver |
| Explainability | `explainability.py` | SHAP values, attention visualization |

See [`docs/ADVANCED_FEATURES.md`](docs/ADVANCED_FEATURES.md) for usage examples.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` or add `--no-amp` flag |
| Graph not connected | Increase `edge_probability` (try 0.1+) |
| Poor accuracy | Train longer (100+ epochs) with more data (5000+ samples) |
| Early stopping too early | Increase `patience` in config |
| Streamlit model not found | Ensure `checkpoints/best_stage1.pt` exists |

---

## � References

- Graph Attention Networks: [Veličković et al., 2018](https://arxiv.org/abs/1710.10903)
- Focal Loss: [Lin et al., 2017](https://arxiv.org/abs/1708.02002)
- Water Distribution Modeling: [EPANET](https://www.epa.gov/water-research/epanet)

---

## 🌍 Real Network Support

This project supports real water distribution network topologies via **EPANET `.inp` files** loaded through the [`wntr`](https://wntr.readthedocs.io/) library.

### Included Network: BattLeDIM L-Town

| Property | Value |
|----------|-------|
| **Nodes** | 785 (782 junctions + 2 reservoirs + 1 tank) |
| **Links** | 909 (905 pipes + 3 valves + 1 pump) |
| **Source** | [BattLeDIM Competition (Zenodo)](https://zenodo.org/records/4017659) |
| **Based on** | Real water network in Cyprus |

The real network pipeline (`dataset_gen_real.py`) generates the **same 7-dim enriched features** as the synthetic pipeline, so the GNN model architecture remains identical.

### Using Your Own Network

```python
from src.utils.graph import load_network_from_inp

G, reservoir_nodes, metadata = load_network_from_inp("path/to/your_network.inp")
print(f"Nodes: {metadata['num_nodes']}, Edges: {metadata['num_edges']}")
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

Made with 💧 by [Gaurav](https://github.com/1Gaurav26)
