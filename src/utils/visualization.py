"""
Visualization utilities for water leak localization.

Provides graph plotting with probability heatmaps, sensor markers,
and leak highlighting using NetworkX and Matplotlib/Plotly.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_graph_with_predictions(
    G: nx.Graph,
    node_probs: np.ndarray,
    sensor_mask: np.ndarray,
    true_leak_node: int = -1,
    reservoir_nodes: Optional[List[int]] = None,
    top_k: int = 5,
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
    title: str = "Water Network - Leak Probability Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot graph with leak probability heatmap using Matplotlib.

    Args:
        G: NetworkX Graph.
        node_probs: Predicted leak probabilities for each node.
        sensor_mask: Binary mask indicating sensor nodes.
        true_leak_node: Index of true leak node (-1 if no leak).
        reservoir_nodes: List of reservoir node indices.
        top_k: Number of top candidates to highlight.
        positions: Node positions for layout.
        title: Plot title.
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get positions
    if positions is None:
        positions = nx.spring_layout(G, seed=42)
    
    # Normalize probabilities for colormap
    prob_min, prob_max = node_probs.min(), node_probs.max()
    if prob_max > prob_min:
        norm_probs = (node_probs - prob_min) / (prob_max - prob_min)
    else:
        norm_probs = np.zeros_like(node_probs)
    
    # Color map for probabilities (white to red)
    cmap = plt.cm.Reds
    node_colors = [cmap(norm_probs[i]) for i in range(len(node_probs))]
    
    # Draw edges
    nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.3, edge_color='gray')
    
    # Draw all nodes with probability colors
    node_list = list(G.nodes())
    nx.draw_networkx_nodes(
        G, positions, ax=ax,
        nodelist=node_list,
        node_color=node_colors,
        node_size=100,
        alpha=0.8,
    )
    
    # Highlight sensor nodes
    sensor_nodes = [i for i, m in enumerate(sensor_mask) if m > 0.5]
    nx.draw_networkx_nodes(
        G, positions, ax=ax,
        nodelist=sensor_nodes,
        node_color='none',
        edgecolors='blue',
        linewidths=2,
        node_size=150,
    )
    
    # Highlight reservoir nodes
    if reservoir_nodes:
        nx.draw_networkx_nodes(
            G, positions, ax=ax,
            nodelist=reservoir_nodes,
            node_color='none',
            edgecolors='green',
            linewidths=3,
            node_size=200,
        )
    
    # Highlight true leak node
    if true_leak_node >= 0:
        nx.draw_networkx_nodes(
            G, positions, ax=ax,
            nodelist=[true_leak_node],
            node_color='none',
            edgecolors='black',
            linewidths=3,
            node_size=300,
            node_shape='*',
        )
    
    # Highlight top-K candidates
    top_k_nodes = np.argsort(node_probs)[-top_k:][::-1]
    for rank, node in enumerate(top_k_nodes):
        x, y = positions[node]
        ax.annotate(
            f"#{rank + 1}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            fontweight='bold',
            color='darkred',
        )
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Leak Probability', rotation=270, labelpad=15)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='blue', markersize=10, markeredgewidth=2,
                   label='Sensor Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                   markeredgecolor='green', markersize=10, markeredgewidth=3,
                   label='Reservoir'),
    ]
    if true_leak_node >= 0:
        legend_elements.append(
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='none',
                       markeredgecolor='black', markersize=15, markeredgewidth=2,
                       label='True Leak')
        )
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_graph_plotly(
    G: nx.Graph,
    node_probs: np.ndarray,
    sensor_mask: np.ndarray,
    true_leak_node: int = -1,
    reservoir_nodes: Optional[List[int]] = None,
    top_k: int = 5,
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
    title: str = "Water Network - Leak Probability Heatmap",
) -> go.Figure:
    """
    Plot interactive graph with Plotly.

    Args:
        G: NetworkX Graph.
        node_probs: Predicted leak probabilities for each node.
        sensor_mask: Binary mask indicating sensor nodes.
        true_leak_node: Index of true leak node (-1 if no leak).
        reservoir_nodes: List of reservoir node indices.
        top_k: Number of top candidates to highlight.
        positions: Node positions for layout.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    if positions is None:
        positions = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Pipes'
    )
    
    # Create node trace
    node_x = [positions[node][0] for node in G.nodes()]
    node_y = [positions[node][1] for node in G.nodes()]
    
    # Node text with info
    node_text = []
    for node in G.nodes():
        text = f"Node {node}<br>Prob: {node_probs[node]:.3f}"
        if sensor_mask[node] > 0.5:
            text += "<br>SENSOR"
        if reservoir_nodes and node in reservoir_nodes:
            text += "<br>RESERVOIR"
        if node == true_leak_node:
            text += "<br>TRUE LEAK"
        node_text.append(text)
    
    # Determine node symbols
    node_symbols = []
    for node in G.nodes():
        if node == true_leak_node:
            node_symbols.append('star')
        elif reservoir_nodes and node in reservoir_nodes:
            node_symbols.append('diamond')
        elif sensor_mask[node] > 0.5:
            node_symbols.append('circle')
        else:
            node_symbols.append('circle')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Reds',
            color=node_probs.tolist(),
            size=10,
            symbol=node_symbols,
            colorbar=dict(
                thickness=15,
                title='Leak Probability',
                xanchor='left',
                titleside='right'
            ),
            line=dict(
                width=[3 if sensor_mask[i] > 0.5 else 0 for i in G.nodes()],
                color='blue'
            )
        ),
        name='Nodes'
    )
    
    # Highlight top-K nodes
    top_k_nodes = np.argsort(node_probs)[-top_k:][::-1]
    top_k_x = [positions[node][0] for node in top_k_nodes]
    top_k_y = [positions[node][1] for node in top_k_nodes]
    top_k_text = [f"#{i+1}: Node {node}<br>Prob: {node_probs[node]:.3f}" 
                  for i, node in enumerate(top_k_nodes)]
    
    top_k_trace = go.Scatter(
        x=top_k_x, y=top_k_y,
        mode='markers+text',
        text=[f"#{i+1}" for i in range(top_k)],
        textposition="top center",
        hoverinfo='text',
        hovertext=top_k_text,
        marker=dict(
            size=20,
            color='rgba(255, 0, 0, 0)',
            line=dict(width=2, color='darkred')
        ),
        name='Top-K Candidates'
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace, top_k_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
        )
    )
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve - Leak Detection",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve for leak detection.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc_score: Area under curve score.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_top_k_bar(
    top_k_accuracies: Dict[int, float],
    title: str = "Top-K Localization Accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart of Top-K accuracies.

    Args:
        top_k_accuracies: Dictionary mapping K to accuracy.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    k_values = list(top_k_accuracies.keys())
    accuracies = list(top_k_accuracies.values())
    
    bars = ax.bar(range(len(k_values)), accuracies, color='steelblue')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"Top-{k}" for k in k_values])
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_magnitude_scatter(
    true_magnitudes: np.ndarray,
    pred_magnitudes: np.ndarray,
    title: str = "Leak Magnitude: Predicted vs True",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot scatter plot of predicted vs true leak magnitudes.

    Args:
        true_magnitudes: Ground truth magnitudes.
        pred_magnitudes: Predicted magnitudes.
        title: Plot title.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(true_magnitudes, pred_magnitudes, alpha=0.5, s=30)
    
    # Add diagonal line (perfect prediction)
    min_val = min(true_magnitudes.min(), pred_magnitudes.min())
    max_val = max(true_magnitudes.max(), pred_magnitudes.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Magnitude')
    ax.set_ylabel('Predicted Magnitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
