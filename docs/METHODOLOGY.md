# Proposed Methodology & Detailed Description of Work Carried Out

## Water Leak Localization in Pipe Networks using Graph Neural Networks and Physics-Based Estimation

---

## 1. Introduction & Problem Statement

Water distribution networks (WDNs) lose **20–30%** of treated water globally due to undetected pipe leaks (WHO, 2022). Traditional leak detection relies on manual acoustic surveys or model-based residual analysis — both are slow, expensive, and scale poorly to networks with hundreds of nodes.

**Our Goal**: Detect *which pipe junction* is leaking and *how much* water is being lost, using only sparse pressure sensor readings across the network.

**Key Challenges**:
- **Extreme class imbalance**: In a 785-node network, only 1 node has a leak → 1:784 positive:negative ratio
- **Sparse observations**: Sensors cover only ~15–20% of all nodes
- **Graph-structured data**: The pipe network has irregular topology that standard CNNs/MLPs cannot exploit
- **Real-time requirement**: Leak localization must be fast enough for operational use

---

## 2. Proposed Methodology

We propose a **two-stage pipeline** that combines graph deep learning with physics-based refinement:

```
Sensor Readings → [Stage 1: GNN Localization] → Top-K Leak Candidates
                                                        ↓
     Leak Location + Magnitude ← [Stage 2: Physics Solver] ←─┘
```

### 2.1 Stage 1: Edge-Gated Multi-Head Attention GNN

#### 2.1.1 Architecture Design

We designed a custom GNN architecture — the **Edge-Gated Attention Network (EGAN)** — specifically tailored for hydraulic networks. Unlike standard Graph Attention Networks (GATs), our model incorporates edge features (pipe conductance) directly into the attention mechanism.

**Architecture Overview**:

| Component | Details |
|-----------|---------|
| **Input Projection** | Linear → BatchNorm1d → GELU → Dropout |
| **Message Passing** | 4 × Edge-Gated Multi-Head Attention Layers |
| **Attention Heads** | 4 heads per layer |
| **Skip Connections** | Residual + LayerNorm across all layers |
| **Node Head** | MLP → per-node leak probability |
| **Global Head** | Attention Pooling → MLP → no-leak score |

**Edge-Gated Attention Mechanism**:

For each edge (i, j) with pipe feature **e**ᵢⱼ (conductance, length), the attention coefficient is computed as:

```
gate_ij = σ(MLP_gate(e_ij))          ← Pipe conductance modulates attention
score_ij = MLP_score([q_i ∥ k_j ∥ e_ij])  ← Learned attention score
α_ij = gate_ij × softmax(score_ij)   ← Gated attention weight
```

This is novel because:
1. **Standard GAT** computes attention only from node features (ignores pipe properties)
2. **Our EGAN** uses the pipe's hydraulic conductance as a learnable gate, allowing the model to weight messages from high-capacity pipes differently than low-capacity ones
3. The per-head gating creates **head specialization** — some heads focus on local neighbors while others attend to structurally important connections

#### 2.1.2 Enriched 7-Dimensional Node Features

Rather than using raw pressure readings alone, we engineer a **7-dimensional feature vector** for each node that combines sensor data with structural graph properties:

| Dim | Feature | Source | Purpose |
|-----|---------|--------|---------|
| 0 | Normalized pressure | Sensor readings | Primary leak signal |
| 1 | Sensor mask | Binary (0/1) | Distinguishes measured vs. interpolated nodes |
| 2 | Node degree (normalized) | Graph topology | Structural importance |
| 3 | Reservoir distance (normalized) | BFS distance | Proximity to water sources |
| 4 | Is reservoir | Binary flag | Boundary condition identification |
| 5 | Mean neighbor conductance | Edge weights | Local pipe capacity context |
| 6 | Pressure deviation | Sensor − global mean | Anomaly detection signal |

**Innovation**: Dimensions 2–5 are *static* (computed once per network) while dimensions 0 and 6 are *dynamic* (change with each leak scenario). This separation allows the GNN to learn both structural priors and leak-specific anomaly patterns.

#### 2.1.3 Loss Function: Focal Loss + Neighborhood Label Smoothing

**Problem**: With 785 nodes and 1 leak, standard Binary Cross-Entropy (BCE) learns to predict "no leak everywhere" because it achieves 99.87% accuracy by doing nothing.

**Our Solution** — Two combined techniques:

**1. Focal Loss (α=0.9, γ=2.0)**:

```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

- α=0.9 upweights the positive (leak) class by 9×
- γ=2.0 downweights easy negatives, focusing on hard examples

**2. Neighborhood Label Smoothing (Novel)**:

Instead of a one-hot label at the leak node, we assign partial labels to its 1-hop graph neighbors:

```
Label at leak node:         1.0
Labels at 1-hop neighbors:  0.3
Labels at all other nodes:  0.0
```

**Why this is novel**: This spreads the gradient signal from 1 node to ~15 nodes (average degree in L-Town). The GNN can now learn that "pressure drops propagate through neighbors" — which is physically correct. This technique is inspired by label smoothing in NLP but adapted to graph topology.

### 2.2 Stage 2: Physics-Based Magnitude Estimation

After Stage 1 identifies the Top-K most likely leak nodes, Stage 2 estimates the exact leak magnitude using a **closed-form physics solver**.

#### 2.2.1 Hydraulic Model

We model the water network as a linear system:

```
Pipe Flow:       Q_ij = g_e × (h_i − h_j)     (conductance × head difference)
Node Continuity: Σ_j Q_ij = d_i               (flow balance at each node)
System Matrix:   L @ h = d                     (weighted Laplacian system)
```

Where **L** is the weighted graph Laplacian constructed from pipe conductances, **h** is the pressure vector, and **d** is the demand vector.

#### 2.2.2 Least-Squares Leak Estimation

For a candidate leak node *k*, the pressure change per unit leak is:

```
Δp = h(d + e_k) − h(d)
```

Where e_k is a unit vector at node k. The optimal leak magnitude is:

```
q* = (Δpᵀ × Δp_obs) / (Δpᵀ × Δp)
```

This is a **closed-form** solution (no iterative optimization), making Stage 2 extremely fast.

#### 2.2.3 Hybrid Refinement

We also implement a **HybridRefiner** that combines:
- Physics-based estimation (primary)
- Learned MLP correction (optional)

The learned component extracts local subgraph features around each candidate and predicts a correction term.

---

## 3. Real Network Integration (BattLeDIM L-Town)

### 3.1 Innovation: Real Topology with Simulated Leaks

Most prior work uses either:
- Fully synthetic random graphs (unrealistic topology), OR
- Real SCADA data with limited labeled leak events (small dataset)

**Our approach combines both**: We load the **real L-Town network topology** (785 nodes, 909 links, based on a real network in Cyprus from the BattLeDIM competition) and simulate thousands of leak scenarios on it.

This gives us:
- **Realistic graph structure** (real pipe diameters, lengths, roughness coefficients)
- **Unlimited labeled training data** (simulated leaks at known locations)
- **Ground truth** for every sample (exact leak node and magnitude)

### 3.2 Innovation: Pressure Residual Features

When transitioning from synthetic grids to real networks, we identified a critical failure mode: **spatial pressure variations eclipsing leak signals**.
- In the L-Town network, elevation and structural differences cause baseline pressures to range from 50m to 3600m across nodes (std = 644m).
- A typical leak causes a pressure drop of only 4m to 40m.
- If feature engineering uses standard global normalization `(observed - global_mean) / global_std`, the leak signal is compressed to ~0.01 standard deviations, making it invisible to the GNN.

**Our Solution**: Instead of absolute pressures, we compute **per-node pressure residuals**:
1. For every sample, we solve the hydraulic system *without* a leak under the current demand.
2. We compute the residual: `residual_i = p_observed_i - p_baseline_i`.
3. The GNN's primary feature is the normalized residual. By subtracting the spatial baseline, the feature vector is perfectly zero-centered and solely represents the anomaly pattern propagating through the graph.

### 3.3 Network Details

| Property | Value |
|----------|-------|
| Junctions | 782 |
| Reservoirs | 2 (R1, R2) |
| Tanks | 1 (T1) |
| Pipes | 905 |
| Valves | 3 |
| Source | BattLeDIM Competition, Zenodo |

### 3.4 Conductance Extraction from EPANET

We extract realistic pipe conductances from the EPANET model. Crucially, we parse **all link types** (pipes, valves, and pumps) to ensure reservoirs remain fully connected to the active hydraulic grid, preventing singular Laplacian matrices. Conductance is estimated as:

```
g = C × D² / L
```

Where C is the Hazen-Williams roughness coefficient, D is pipe diameter, and L is pipe length. These are then normalized to [0.5, 2.0] for numerical stability.

---

## 4. Training Pipeline (Production-Grade)

### 4.1 Training Enhancements

| Feature | Details | Purpose |
|---------|---------|---------|
| **Optimizer** | AdamW (weight decay 1e-4) | Regularization via decoupled weight decay |
| **LR Schedule** | Cosine annealing + linear warmup (5–8 epochs) | Stable convergence |
| **Mixed Precision** | `torch.amp` on CUDA | 2× faster training on GPU |
| **Gradient Accumulation** | 4 steps (effective batch=128) | Larger effective batch on limited memory |
| **Model Selection** | Combined score: 0.4 × AUC + 0.6 × Top-5 | Balances detection and localization |
| **Early Stopping** | Patience = 15–20 epochs | Prevents overfitting |

### 4.2 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **ROC-AUC** | Leak detection accuracy (binary: leak vs no-leak) |
| **Top-1 Accuracy** | True leak node is the #1 prediction |
| **Top-3 Accuracy** | True leak node is within top 3 predictions |
| **Top-5 Accuracy** | True leak node is within top 5 predictions |
| **Mean Graph Distance** | Shortest path hops from prediction to true leak |
| **Magnitude MAE** | Mean absolute error of estimated leak magnitude |

---

## 5. Baselines for Comparison

We implement three baselines to benchmarking our GNN approach:

### 5.1 Pressure Residual Ranking
Ranks nodes by correlation between observed pressure residual and the sensitivity pattern of each candidate node. Uses the sensitivity matrix from the hydraulic model.

### 5.2 Graph Distance Baseline
Ranks nodes by proximity (in graph distance) to the sensor showing the largest pressure deviation.

### 5.3 MLP Baseline
Flattens the graph features and applies a standard Multi-Layer Perceptron, ignoring graph structure.

---

## 6. Innovation & Novelty Summary

| # | Innovation | Existing Approaches | Our Contribution |
|---|-----------|-------------------|------------------|
| 1 | **Edge-Gated Attention** | Standard GAT ignores edge attributes | Per-head conductance gating learns pipe-aware message passing |
| 2 | **Neighborhood Label Smoothing** | One-hot labels at leak node only | Partial labels at 1-hop neighbors exploit physics of pressure propagation |
| 3 | **7-Dim Enriched Features** | Raw pressure only (1–2 dim) | Structural + dynamic features give GNN geometry awareness |
| 4 | **Two-Stage Hybrid Architecture** | End-to-end ML OR pure physics | GNN localization + physics-based magnitude estimation |
| 5 | **Real Topology + Simulated Leaks** | Random graphs OR limited real data | BattLeDIM L-Town topology with unlimited synthetic scenarios |
| 6 | **Production Training Pipeline** | Basic training loops | AMP, gradient accumulation, cosine warmup, focal loss |

### Key Differentiators from Prior Work

1. **Veličković et al. (GAT, 2018)**: Original GAT does not use edge features. Our edge-gated attention incorporates pipe conductance as a learnable gate.

2. **BattLeDIM Competition Methods**: Most competition entries use model-based or statistical methods. We combine graph deep learning with physics for a hybrid approach.

3. **Standard GNN for WDN**: Previous GNN-based leak detection work typically uses 1–2 dimensional features and standard message passing. We use 7-dimensional enriched features and edge-gated multi-head attention.

---

## 7. Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+ |
| **Graph Operations** | NetworkX |
| **Hydraulic Modeling** | Custom Laplacian solver + wntr (EPANET) |
| **Numerical Computing** | NumPy, SciPy |
| **Visualization** | Matplotlib, Plotly |
| **Web Demo** | Streamlit |
| **Configuration** | YAML + dataclasses |

---

## 8. References

1. Veličković, P., et al. "Graph Attention Networks." ICLR 2018.
2. Lin, T.-Y., et al. "Focal Loss for Dense Object Detection." ICCV 2017.
3. Vrachimis, S. G., et al. "BattLeDIM: Battle of the Leakage Detection and Isolation Methods." ASCE 2022.
4. Rossman, L. A. "EPANET 2 — Users Manual." EPA 2000.
5. Klise, K. A., et al. "Water Network Tool for Resilience (WNTR)." Sandia National Laboratories 2017.
