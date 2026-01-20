# Water Leak Localization in Pipe Networks using Graph ML + Synthetic Leaks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete machine learning pipeline for detecting and localizing water leaks in pipe networks using Graph Neural Networks (GNNs) and synthetic leak simulation.

**🚀 NEW: Advanced Patentable Features Now Available!**

This repository now includes 8 advanced features for state-of-the-art leak detection:
- 🎯 Multi-leak detection with combinatorial optimization
- ⏱️ Temporal-aware leak evolution tracking
- 📊 Bayesian GNN with uncertainty quantification
- 📍 Adaptive sensor placement optimization
- 🔀 Multi-modal sensor fusion (pressure + flow + acoustic)
- 🔄 Transfer learning across networks
- ⚙️ Physics-Informed Neural Networks (PINN)
- 🔍 Explainable AI with attention visualization

![Demo Preview](results/comparison.png)

## 🎯 Overview

This repository implements a comprehensive two-stage approach for water leak localization, enhanced with advanced machine learning techniques:

1. **Stage 1: GNN-based Detection & Localization**
   - Graph Neural Network processes sparse sensor readings
   - Outputs per-node leak probability heatmap
   - Global no-leak score for leak detection
   - **NEW**: Multi-leak detection, temporal analysis, and uncertainty quantification

2. **Stage 2: Physics-based Magnitude Estimation**
   - Uses closed-form linear algebra solution
   - Minimizes sensor residual error to estimate leak magnitude
   - **NEW**: Physics-Informed Neural Network hybrid solver

## 📋 Features

### Core Features
- ✅ Linear hydraulic simulator for water networks
- ✅ Synthetic dataset generation with configurable parameters
- ✅ Manual message-passing GNN (no PyTorch Geometric required)
- ✅ Multiple baseline methods for comparison
- ✅ Comprehensive evaluation metrics
- ✅ Interactive Streamlit demo application
- ✅ YAML-based configuration system
- ✅ Deterministic training with random seeds

### 🚀 Advanced Features (Patentable Innovations)

- **Multi-Leak Detection**: Simultaneous detection of multiple concurrent leaks using combinatorial optimization
- **Temporal Analysis**: Time-series GNN with attention for leak evolution tracking and early warning
- **Uncertainty Quantification**: Bayesian GNN with confidence intervals (epistemic + aleatoric uncertainty)
- **Sensor Placement**: Adaptive optimization using information gain and graph centrality
- **Multi-Modal Fusion**: Cross-modal attention for pressure, flow, and acoustic sensors
- **Transfer Learning**: Domain adaptation for cross-network model transfer
- **Physics-Informed NN**: Hybrid solver combining learned and physics-based methods
- **Explainability**: Attention visualization, SHAP values, and causal path analysis

See [`docs/ADVANCED_FEATURES.md`](docs/ADVANCED_FEATURES.md) and [`docs/PATENT_STRATEGY.md`](docs/PATENT_STRATEGY.md) for details.

## 🏗️ Project Structure

```
├── src/
│   ├── sim/
│   │   ├── simulator.py      # Linear hydraulic simulator
│   │   └── dataset_gen.py    # Dataset generation
│   ├── models/
│   │   ├── gnn_stage1.py          # Stage 1 GNN model
│   │   ├── stage2_refine.py       # Stage 2 magnitude estimation
│   │   ├── gnn_multileak.py       # ✨ Multi-leak detection
│   │   ├── gnn_temporal.py        # ✨ Temporal-aware GNN
│   │   ├── gnn_bayesian.py        # ✨ Bayesian GNN with uncertainty
│   │   ├── sensor_placement.py    # ✨ Sensor placement optimization
│   │   ├── multimodal_fusion.py   # ✨ Multi-modal fusion
│   │   ├── transfer_learning.py   # ✨ Transfer learning
│   │   └── pinn_hybrid.py         # ✨ Physics-Informed NN
│   ├── baselines/
│   │   ├── residual_baseline.py     # Pressure residual ranking
│   │   ├── graph_distance_baseline.py  # Graph distance method
│   │   └── mlp_baseline.py          # Simple MLP baseline
│   └── utils/
│       ├── config.py         # Configuration management
│       ├── seed.py           # Random seed utilities
│       ├── graph.py          # Graph utilities
│       ├── metrics.py        # Evaluation metrics
│       ├── visualization.py  # Plotting utilities
│       └── explainability.py # ✨ Explainability module (SHAP, attention)
├── scripts/
│   ├── generate_data.py      # Data generation script
│   ├── train_stage1.py       # Stage 1 training script
│   ├── train_stage2.py       # Stage 2 calibration script
│   └── evaluate.py           # Evaluation script
├── app/
│   └── streamlit_app.py      # Interactive demo
├── configs/
│   └── default.yaml          # Default configuration
├── docs/
│   ├── PATENT_STRATEGY.md    # ✨ Patent filing strategy
│   ├── ADVANCED_FEATURES.md  # ✨ Advanced features documentation
│   └── IMPLEMENTATION_COMPLETE.md  # ✨ Implementation status
├── requirements.txt
└── README.md
```

✨ = Advanced patentable features

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/1Gaurav26/Tarp.git
cd Tarp

# Create virtual environment (optional but recommended)
python -m venv tarp
source tarp/bin/activate  # On Windows: tarp\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate train/val/test datasets with default settings
python scripts/generate_data.py

# Or customize parameters
python scripts/generate_data.py --num-nodes 100 --train-samples 1000 --seed 42
```

### 3. Train the Model

```bash
# Train Stage 1 GNN
python scripts/train_stage1.py --epochs 50

# Validate Stage 2 magnitude estimation
python scripts/train_stage2.py
```

### 4. Evaluate

```bash
# Run full evaluation
python scripts/evaluate.py

# Results saved to results/ directory
```

### 5. Run the Demo

```bash
# Launch Streamlit app
streamlit run app/streamlit_app.py
```

## ⚙️ Configuration

All parameters are configured via `configs/default.yaml`:

```yaml
# Graph Parameters
graph:
  num_nodes: 200
  edge_probability: 0.08
  reservoir_nodes: 2

# Sensor Configuration
sensors:
  ratio: 0.2              # 20% of nodes are sensors
  noise_std: 0.1          # Gaussian noise level

# Leak Parameters
leak:
  probability: 0.8        # 80% of samples have leaks
  magnitude_min: 0.5
  magnitude_max: 5.0

# Training
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001

# Model
model:
  hidden_dim: 64
  num_layers: 3

# Reproducibility
seed: 42
```

## 🔬 Simulator Equations

The linear hydraulic simulator uses the following equations:

### Node Pressure (Head)
- Reservoir nodes have fixed head: `h_reservoir = H0` (default: 50m)
- Other node heads are solved from the linear system

### Pipe Flow
```
Q_ij = g_e * (h_i - h_j)
```
where `g_e = 1/R_e` is the pipe conductance.

### Node Continuity
```
Σ_j Q_ij = d_i
```
where `d_i` is the demand at node `i` (positive = outflow).

### System Matrix
The weighted Laplacian `L` is assembled from conductances:
```
L @ h = d
```
with boundary conditions at reservoir nodes.

### Leak Injection
A leak at node `k` with magnitude `q` is modeled as additional demand:
```
d_k' = d_k + q
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Leak detection (leak vs. no-leak) |
| **Top-1 Accuracy** | True leak in top-1 prediction |
| **Top-K Accuracy** | True leak in top-K predictions |
| **Graph Distance** | Shortest path from prediction to true leak |
| **Magnitude MAE** | Mean absolute error of leak magnitude |

## 📈 Expected Results

With default settings (200 nodes, 20% sensors, 2000 training samples):

| Method | Top-1 | Top-5 | ROC-AUC |
|--------|-------|-------|---------|
| **GNN** | ~0.40 | ~0.70 | ~0.85 |
| Residual Baseline | ~0.25 | ~0.50 | - |
| Graph Distance | ~0.15 | ~0.35 | - |

*Results vary with random seeds and configurations.*

## 🖥️ Streamlit Demo Features

The interactive demo allows you to:

- 🎲 Generate random leak scenarios
- 🎯 Select specific leak node and magnitude
- 📊 Adjust sensor count and noise level
- 🗺️ Visualize network with probability heatmap
- 📈 View top-K candidates and predictions
- 📉 Display precomputed ROC curves and metrics

## 🧪 Running Tests

```bash
# Quick test with small dataset
python scripts/generate_data.py --num-nodes 50 --train-samples 100

# Fast training test
python scripts/train_stage1.py --epochs 5 --batch-size 16

# Evaluate
python scripts/evaluate.py
```

## 📦 Data Format

Each `.pt` file contains:

```python
{
    "node_features": [N, num_nodes, 2],      # (pressure, sensor_mask)
    "edge_index": [2, num_edges*2],          # Bidirectional edges
    "edge_attr": [num_edges*2, 2],           # (conductance, length)
    "y_node": [N, num_nodes],                # One-hot leak labels
    "has_leak": [N],                         # Binary leak indicator
    "leak_nodes": [N],                       # Leak node indices
    "leak_magnitudes": [N],                  # Leak magnitudes
    "sensor_mask": [num_nodes],              # Sensor node mask
    "full_pressures": [N, num_nodes],        # True pressure field
    "demands": [N, num_nodes],               # Demand vectors
}
```

## 🔧 Advanced Usage

### Custom Graph Generation

```python
from src.utils.graph import create_random_graph, create_grid_graph

# Random Erdos-Renyi graph
G = create_random_graph(num_nodes=200, edge_probability=0.08)

# Grid-like graph (common in water networks)
G = create_grid_graph(rows=15, cols=15, add_diagonals=True)
```

### 🚀 Advanced Features Usage

#### Multi-Leak Detection

```python
from src.models.gnn_multileak import MultiLeakDetectionGNN

model = MultiLeakDetectionGNN(max_leaks=3)
leak_nodes, probs, score = model.predict_multi_leak_combinatorial(
    node_probs, num_leaks=2, G=graph
)
```

#### Temporal-Aware Detection

```python
from src.models.gnn_temporal import TemporalGNN

model = TemporalGNN()
predictions = model.predict_with_evolution(
    x_seq, edge_index, edge_attr, forecast_steps=5
)
```

#### Uncertainty Quantification

```python
from src.models.gnn_bayesian import BayesianLeakDetectionGNN

model = BayesianLeakDetectionGNN(use_bayesian=True, mc_samples=10)
predictions = model.predict_with_uncertainty(
    x, edge_index, edge_attr, num_samples=10
)
# Access: predictions["mean"], predictions["confidence_intervals"]
```

#### Sensor Placement Optimization

```python
from src.models.sensor_placement import SensorPlacementOptimizer

optimizer = SensorPlacementOptimizer(G, simulator)
sensor_nodes, metrics = optimizer.optimize_placement(
    num_sensors=20, method="hybrid"
)
```

#### Multi-Modal Fusion

```python
from src.models.multimodal_fusion import MultiModalFusionGNN

model = MultiModalFusionGNN(pressure_dim=2, flow_dim=1, acoustic_dim=5)
node_probs, no_leak_prob, attention = model.predict(
    pressure_features, flow_features, acoustic_features,
    edge_index=edge_index, edge_attr=edge_attr
)
```

#### Transfer Learning

```python
from src.models.transfer_learning import TransferableLeakDetectionGNN

model = TransferableLeakDetectionGNN(base_gnn=pretrained_model)
model.fine_tune_on_target(target_data, epochs=10, freeze_base=True)
```

#### Physics-Informed NN

```python
from src.models.pinn_hybrid import PINNLeakDetectionGNN

model = PINNLeakDetectionGNN(simulator=simulator, G=graph)
results = model.hybrid_predict(x, edge_index, edge_attr, demands)
```

#### Explainability

```python
from src.utils.explainability import AttentionVisualizer, ExplanationGenerator

visualizer = AttentionVisualizer(G)
fig = visualizer.visualize_node_attention(attention_weights)

explainer = ExplanationGenerator(G, simulator)
explanation = explainer.generate_explanation(
    predicted_leak_node, leak_prob, top_k_nodes
)
```

For more examples, see [`docs/ADVANCED_FEATURES.md`](docs/ADVANCED_FEATURES.md).

### Using the Simulator

```python
from src.sim.simulator import HydraulicSimulator

simulator = HydraulicSimulator(
    G=graph,
    reservoir_nodes=[0, 1],
    reservoir_head=50.0,
)

# Generate demand
demand = simulator.generate_demand(mean=1.0, std=0.5)

# Solve with leak
heads = simulator.solve(demand, leak_node=42, leak_magnitude=3.0)

# Get sensor readings
readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std=0.1)
```

### Physics-based Magnitude Estimation

```python
from src.models.stage2_refine import PhysicsBasedRefiner

refiner = PhysicsBasedRefiner(simulator, sensor_nodes)

# Estimate magnitude for top candidates
best_node, best_magnitude, all_estimates = refiner.refine_top_k(
    top_k_nodes=[24, 56, 78],
    observed_pressures=sensor_readings,
    base_demand=demand,
)
```

## 📚 Documentation

- **[Advanced Features Guide](docs/ADVANCED_FEATURES.md)** - Detailed documentation of all advanced features
- **[Patent Strategy](docs/PATENT_STRATEGY.md)** - Patent filing strategy and innovations
- **[Implementation Status](docs/IMPLEMENTATION_COMPLETE.md)** - Complete implementation summary

## 📚 References

- Graph Neural Networks for water networks: [Link]
- Water distribution system modeling: [EPANET](https://www.epa.gov/water-research/epanet)
- Leak detection methods: [Survey paper]

## 🎯 Patentable Innovations

This repository includes **8 advanced patentable features** for state-of-the-art leak detection:

1. **Multi-Leak Detection** - Combinatorial optimization for simultaneous leak detection
2. **Temporal Analysis** - Time-series GNN for leak evolution tracking
3. **Uncertainty Quantification** - Bayesian GNN with confidence intervals
4. **Sensor Placement** - Information-theoretic optimization
5. **Multi-Modal Fusion** - Cross-modal attention for multiple sensor types
6. **Transfer Learning** - Domain adaptation across networks
7. **Physics-Informed NN** - Hybrid physics+learned solver
8. **Explainability** - Attention visualization and SHAP analysis

See [`docs/PATENT_STRATEGY.md`](docs/PATENT_STRATEGY.md) for detailed patent filing strategy.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or use CPU
2. **Graph not connected**: Increase `edge_probability`
3. **Poor performance**: Try more training epochs or larger dataset
4. **Streamlit not finding model**: Ensure checkpoint path is correct

### Getting Help

If you encounter any issues, please:
1. Check the [Issues](https://github.com/yourusername/water-leak-localization/issues) page
2. Create a new issue with your error message and configuration

---

## 📊 Project Status

✅ **Core Features**: Fully implemented and tested  
🚀 **Advanced Features**: All 8 patentable features implemented  
📄 **Documentation**: Complete with patent strategy  
🧪 **Testing**: Unit tests and integration tests available  

**Latest Update**: All advanced patentable features successfully implemented!

---

Made with 💧 by Your Team
