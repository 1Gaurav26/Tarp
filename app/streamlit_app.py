#!/usr/bin/env python3
"""
Streamlit Demo App for Water Leak Localization (Production).

Interactive visualization of leak detection and localization in pipe networks.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import networkx as nx

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.graph import create_random_graph, select_sensor_nodes, select_reservoir_nodes, get_node_positions
from src.utils.visualization import plot_graph_plotly
from src.sim.simulator import HydraulicSimulator
from src.sim.dataset_gen import LeakDataset, _compute_static_features
from src.models.gnn_stage1 import LeakDetectionGNN
from src.models.stage2_refine import PhysicsBasedRefiner


# Page config
st.set_page_config(
    page_title="Water Leak Localization",
    page_icon="💧",
    layout="wide",
)


@st.cache_resource
def load_model_and_data(config_path: str, checkpoint_path: str, data_path: str):
    """Load model and data (cached)."""
    config = load_config(config_path)
    device = "cpu"

    # Load dataset for graph structure
    if Path(data_path).exists():
        dataset = LeakDataset(data_path)
        G = dataset.get_graph()
        num_nodes = dataset.num_nodes
        reservoir_nodes = dataset.reservoir_nodes
        sensor_nodes = dataset.data["sensor_nodes"]
        conductances = dataset.data["conductances"].numpy()
        positions = dataset.data.get("positions", None)
        in_dim = dataset.node_features.shape[-1]
    else:
        set_seed(config.seed)
        G = create_random_graph(config.graph.num_nodes, config.graph.edge_probability, seed=config.seed)
        num_nodes = config.graph.num_nodes
        reservoir_nodes = select_reservoir_nodes(G, config.graph.reservoir_nodes, seed=config.seed)
        sensor_nodes = select_sensor_nodes(G, config.sensors.ratio, exclude_nodes=reservoir_nodes, seed=config.seed)
        conductances = np.random.uniform(config.graph.min_conductance, config.graph.max_conductance, size=G.number_of_edges())
        positions = get_node_positions(G)
        dataset = None
        in_dim = config.model.in_dim

    # Create simulator
    simulator = HydraulicSimulator(
        G=G,
        reservoir_nodes=reservoir_nodes,
        reservoir_head=config.demand.reservoir_head,
        conductances=conductances,
        seed=config.seed,
    )

    # Load model
    model = None
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        edge_dim = 2

        model = LeakDetectionGNN(
            in_dim=in_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            edge_dim=edge_dim,
            num_heads=config.model.num_heads,
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    return config, G, simulator, sensor_nodes, reservoir_nodes, positions, model, dataset


def create_sample(
    G: nx.Graph,
    simulator: HydraulicSimulator,
    sensor_nodes: list,
    reservoir_nodes: list,
    leak_node: int,
    leak_magnitude: float,
    noise_std: float,
    demand_mean: float = 1.0,
    demand_std: float = 0.5,
    seed: int = None,
):
    """Create a sample scenario with enriched 7-dim features."""
    if seed is not None:
        np.random.seed(seed)

    num_nodes = G.number_of_nodes()
    demand = simulator.generate_demand(mean=demand_mean, std=demand_std)

    has_leak = leak_node >= 0
    heads = simulator.solve(
        demand,
        leak_node=leak_node if has_leak else None,
        leak_magnitude=leak_magnitude if has_leak else 0.0,
    )

    sensor_readings = simulator.get_sensor_readings(heads, sensor_nodes, noise_std)

    sensor_mask = np.zeros(num_nodes, dtype=np.float32)
    sensor_mask[sensor_nodes] = 1.0

    # Compute static features
    static_feats = _compute_static_features(
        G, num_nodes, sensor_mask, list(reservoir_nodes) if not isinstance(reservoir_nodes, list) else reservoir_nodes,
        simulator.edge_conductance,
    )

    # Baseline for normalization
    baseline_demand = np.full(num_nodes, demand_mean, dtype=np.float64)
    for r in reservoir_nodes:
        baseline_demand[r] = 0.0
    baseline_heads = simulator.solve(baseline_demand)
    pressure_mean = baseline_heads.mean()
    pressure_std = baseline_heads.std() if baseline_heads.std() > 0 else 1.0

    # Build enriched 7-dim node features
    node_features = np.zeros((num_nodes, 7), dtype=np.float32)

    for idx, s_node in enumerate(sensor_nodes):
        node_features[s_node, 0] = (sensor_readings[idx] - pressure_mean) / pressure_std

    node_features[:, 1] = sensor_mask
    node_features[:, 2:6] = static_feats

    if len(sensor_readings) > 0:
        global_sensor_mean = sensor_readings.mean()
        for idx, s_node in enumerate(sensor_nodes):
            node_features[s_node, 6] = (sensor_readings[idx] - global_sensor_mean) / pressure_std

    edge_index, edge_attr = simulator.get_edge_features()

    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "sensor_mask": torch.tensor(sensor_mask, dtype=torch.float32),
        "demand": demand,
        "heads": heads,
        "sensor_readings": sensor_readings,
    }


def main():
    st.title("💧 Water Leak Localization")
    st.markdown("Interactive demo for detecting and localizing water leaks using Graph Neural Networks.")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    config_path = st.sidebar.text_input("Config Path", value="configs/default.yaml")
    checkpoint_path = st.sidebar.text_input("Model Checkpoint", value="checkpoints/best_stage1.pt")
    data_path = st.sidebar.text_input("Data Path", value="data/test.pt")

    try:
        config, G, simulator, sensor_nodes, reservoir_nodes, positions, model, dataset = \
            load_model_and_data(config_path, checkpoint_path, data_path)
        st.sidebar.success("✅ Resources loaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
        st.stop()

    num_nodes = G.number_of_nodes()

    st.sidebar.header("🔧 Scenario")
    scenario_mode = st.sidebar.radio("Mode", ["Random Leak", "Custom Leak", "No Leak"])

    if scenario_mode == "Random Leak":
        seed = st.sidebar.number_input("Seed", value=42, min_value=0)
        np.random.seed(seed)
        leak_candidates = [n for n in range(num_nodes) if n not in reservoir_nodes]
        leak_node = np.random.choice(leak_candidates)
        leak_magnitude = np.random.uniform(config.leak.magnitude_min, config.leak.magnitude_max)
    elif scenario_mode == "Custom Leak":
        leak_candidates = [n for n in range(num_nodes) if n not in reservoir_nodes]
        leak_node = st.sidebar.selectbox("Leak Node", leak_candidates)
        leak_magnitude = st.sidebar.slider("Magnitude", 0.1, 10.0, 2.0, 0.1)
        seed = st.sidebar.number_input("Seed", value=42, min_value=0)
    else:
        leak_node = -1
        leak_magnitude = 0.0
        seed = st.sidebar.number_input("Seed", value=42, min_value=0)

    noise_std = st.sidebar.slider("Noise Level", 0.0, 1.0, float(config.sensors.noise_std), 0.05)
    top_k = st.sidebar.slider("Top-K Candidates", 1, 20, 5)

    # Create sample
    sample = create_sample(G, simulator, sensor_nodes, reservoir_nodes, leak_node, leak_magnitude, noise_std, seed=seed)

    # Inference
    if model is not None:
        with torch.no_grad():
            node_probs, no_leak_prob = model.predict(
                sample["node_features"].unsqueeze(0),
                sample["edge_index"],
                sample["edge_attr"],
            )
            node_probs = node_probs.squeeze(0).numpy()
            has_leak_prob = 1 - no_leak_prob.item()
    else:
        node_probs = np.ones(num_nodes) / num_nodes
        has_leak_prob = 0.5
        st.warning("⚠️ No trained model found. Using uniform predictions.")

    # Display
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Network Visualization")
        sensor_mask = sample["sensor_mask"].numpy()
        fig = plot_graph_plotly(
            G, node_probs, sensor_mask,
            true_leak_node=leak_node,
            reservoir_nodes=reservoir_nodes,
            top_k=top_k,
            positions=positions,
            title="Pipe Network - Leak Probability Heatmap",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("📊 Predictions")

        st.markdown("### Leak Detection")
        if scenario_mode != "No Leak":
            st.markdown(f"**True Status:** 🔴 Leak at Node {leak_node}")
            st.markdown(f"**True Magnitude:** {leak_magnitude:.2f}")
        else:
            st.markdown("**True Status:** 🟢 No Leak")

        st.markdown(f"**Predicted Leak Probability:** {has_leak_prob:.2%}")
        if has_leak_prob > 0.5:
            st.success("Model predicts: LEAK DETECTED")
        else:
            st.info("Model predicts: NO LEAK")

        st.markdown("### Top-K Candidates")
        top_k_nodes = np.argsort(node_probs)[-top_k:][::-1]
        for rank, node in enumerate(top_k_nodes, 1):
            prob = node_probs[node]
            marker = "✅" if node == leak_node else ""
            st.markdown(f"**#{rank}** Node {node}: {prob:.4f} {marker}")

        if scenario_mode != "No Leak" and model is not None:
            st.markdown("### Magnitude Estimation")
            refiner = PhysicsBasedRefiner(simulator, sensor_nodes)
            try:
                best_node, best_magnitude, _ = refiner.refine_top_k(
                    top_k_nodes.tolist(), sample["sensor_readings"], sample["demand"],
                )
                st.markdown(f"**Predicted Node:** {best_node}")
                st.markdown(f"**Predicted Magnitude:** {best_magnitude:.2f}")
                if best_node == leak_node:
                    st.markdown(f"**Magnitude Error:** {abs(best_magnitude - leak_magnitude):.2f}")
            except Exception as e:
                st.error(f"Magnitude estimation failed: {e}")

    # Precomputed metrics
    if Path("results/evaluation_results.pt").exists():
        st.markdown("---")
        st.subheader("📈 Precomputed Metrics")
        results = torch.load("results/evaluation_results.pt", weights_only=False)

        col1, col2, col3 = st.columns(3)
        if "gnn" in results and results["gnn"]:
            with col1:
                st.metric("ROC-AUC", f"{results['gnn'].get('roc_auc', 0):.4f}")
            with col2:
                st.metric("Top-1 Accuracy", f"{results['gnn'].get('top_1_accuracy', 0):.4f}")
            with col3:
                st.metric("Top-5 Accuracy", f"{results['gnn'].get('top_5_accuracy', 0):.4f}")

        if Path("results/roc_curve.png").exists():
            st.image("results/roc_curve.png", caption="ROC Curve - Leak Detection")
        if Path("results/comparison.png").exists():
            st.image("results/comparison.png", caption="Method Comparison")

    st.markdown("---")
    with st.expander("ℹ️ About This Demo"):
        st.markdown("""
        ### Water Leak Localization using Graph ML

        **Stage 1: GNN with Edge-Gated Attention**
        - Multi-head attention with edge gating by pipe conductance
        - 7-dimensional enriched node features
        - Per-node leak probability heatmap + global no-leak score

        **Stage 2: Physics-based Magnitude Estimation**
        - Closed-form least-squares solution
        - Uses sensor residuals to estimate leak magnitude

        **Visualization Guide:**
        - 🔵 Blue rings → Sensor nodes
        - 🟢 Green diamonds → Reservoir nodes
        - ⭐ Star → True leak location
        - Color intensity → Predicted leak probability
        """)


if __name__ == "__main__":
    main()
