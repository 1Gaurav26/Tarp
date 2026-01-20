"""
Evaluation metrics for water leak localization.

Provides metrics for:
- Leak detection (ROC-AUC)
- Leak localization (Top-K accuracy)
- Graph distance error
- Magnitude estimation (MAE)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import networkx as nx


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute ROC-AUC score for leak detection.

    Args:
        y_true: Binary labels (1 = leak, 0 = no leak).
        y_pred: Predicted probabilities.

    Returns:
        ROC-AUC score.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5  # Cannot compute AUC with single class
    
    return roc_auc_score(y_true, y_pred)


def compute_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for visualization.

    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.

    Returns:
        Tuple of (fpr, tpr, thresholds).
    """
    return roc_curve(y_true, y_pred)


def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute precision-recall curve and average precision.

    Args:
        y_true: Binary labels.
        y_pred: Predicted probabilities.

    Returns:
        Tuple of (precision, recall, average_precision).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return precision, recall, ap


def compute_top_k_accuracy(
    true_node: int,
    predicted_probs: np.ndarray,
    k: int,
) -> bool:
    """
    Check if true leak node is in top-K predictions.

    Args:
        true_node: Ground truth leak node index.
        predicted_probs: Predicted probabilities for each node.
        k: Number of top candidates to consider.

    Returns:
        True if true_node is in top-K predictions.
    """
    if true_node < 0:  # No leak case
        return False
    
    top_k_nodes = np.argsort(predicted_probs)[-k:][::-1]
    return true_node in top_k_nodes


def compute_top_k_accuracy_batch(
    true_nodes: np.ndarray,
    predicted_probs: np.ndarray,
    k: int,
) -> float:
    """
    Compute Top-K accuracy over a batch of samples.

    Args:
        true_nodes: Ground truth leak node indices (-1 for no leak).
        predicted_probs: Predicted probabilities [batch_size, num_nodes].
        k: Number of top candidates.

    Returns:
        Top-K accuracy (fraction of samples where true node is in top-K).
    """
    correct = 0
    total = 0
    
    for i, true_node in enumerate(true_nodes):
        if true_node >= 0:  # Only count samples with leaks
            if compute_top_k_accuracy(true_node, predicted_probs[i], k):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def compute_graph_distance_error(
    G: nx.Graph,
    true_node: int,
    predicted_node: int,
) -> int:
    """
    Compute graph distance between true and predicted leak nodes.

    Args:
        G: NetworkX Graph.
        true_node: Ground truth leak node.
        predicted_node: Predicted leak node (top-1).

    Returns:
        Shortest path length between nodes.
    """
    if true_node < 0 or predicted_node < 0:
        return -1
    
    try:
        return nx.shortest_path_length(G, true_node, predicted_node)
    except nx.NetworkXNoPath:
        return -1


def compute_mean_graph_distance_error(
    G: nx.Graph,
    true_nodes: np.ndarray,
    predicted_probs: np.ndarray,
) -> float:
    """
    Compute mean graph distance error over samples.

    Args:
        G: NetworkX Graph.
        true_nodes: Ground truth leak node indices.
        predicted_probs: Predicted probabilities [batch_size, num_nodes].

    Returns:
        Mean graph distance error (only for samples with leaks).
    """
    errors = []
    
    for i, true_node in enumerate(true_nodes):
        if true_node >= 0:
            predicted_node = np.argmax(predicted_probs[i])
            error = compute_graph_distance_error(G, true_node, predicted_node)
            if error >= 0:
                errors.append(error)
    
    return np.mean(errors) if errors else 0.0


def compute_magnitude_mae(
    true_magnitudes: np.ndarray,
    predicted_magnitudes: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Mean Absolute Error for leak magnitude prediction.

    Args:
        true_magnitudes: Ground truth leak magnitudes.
        predicted_magnitudes: Predicted leak magnitudes.
        mask: Optional mask for samples to include (e.g., only leaky samples).

    Returns:
        MAE for leak magnitude.
    """
    if mask is not None:
        true_magnitudes = true_magnitudes[mask]
        predicted_magnitudes = predicted_magnitudes[mask]
    
    if len(true_magnitudes) == 0:
        return 0.0
    
    return np.mean(np.abs(true_magnitudes - predicted_magnitudes))


def compute_magnitude_rmse(
    true_magnitudes: np.ndarray,
    predicted_magnitudes: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Root Mean Square Error for leak magnitude prediction.

    Args:
        true_magnitudes: Ground truth leak magnitudes.
        predicted_magnitudes: Predicted leak magnitudes.
        mask: Optional mask for samples to include.

    Returns:
        RMSE for leak magnitude.
    """
    if mask is not None:
        true_magnitudes = true_magnitudes[mask]
        predicted_magnitudes = predicted_magnitudes[mask]
    
    if len(true_magnitudes) == 0:
        return 0.0
    
    return np.sqrt(np.mean((true_magnitudes - predicted_magnitudes) ** 2))


def compute_all_metrics(
    G: nx.Graph,
    has_leak: np.ndarray,
    has_leak_pred: np.ndarray,
    true_nodes: np.ndarray,
    node_probs: np.ndarray,
    true_magnitudes: np.ndarray,
    pred_magnitudes: np.ndarray,
    top_k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        G: NetworkX Graph.
        has_leak: Ground truth leak presence (binary).
        has_leak_pred: Predicted leak presence probability.
        true_nodes: Ground truth leak node indices.
        node_probs: Predicted node probabilities.
        true_magnitudes: Ground truth leak magnitudes.
        pred_magnitudes: Predicted leak magnitudes.
        top_k_values: List of K values for Top-K accuracy.

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {}
    
    # Detection metrics
    metrics["roc_auc"] = compute_roc_auc(has_leak, has_leak_pred)
    
    # Localization metrics
    for k in top_k_values:
        metrics[f"top_{k}_accuracy"] = compute_top_k_accuracy_batch(
            true_nodes, node_probs, k
        )
    
    # Graph distance error
    metrics["mean_graph_distance"] = compute_mean_graph_distance_error(
        G, true_nodes, node_probs
    )
    
    # Magnitude metrics (only for samples with leaks)
    leak_mask = has_leak > 0.5
    metrics["magnitude_mae"] = compute_magnitude_mae(
        true_magnitudes, pred_magnitudes, leak_mask
    )
    metrics["magnitude_rmse"] = compute_magnitude_rmse(
        true_magnitudes, pred_magnitudes, leak_mask
    )
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metric names to values.

    Returns:
        Formatted string.
    """
    lines = ["=" * 50, "Evaluation Metrics", "=" * 50]
    
    lines.append("\n--- Leak Detection ---")
    lines.append(f"ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}")
    
    lines.append("\n--- Leak Localization ---")
    for key in sorted(metrics.keys()):
        if key.startswith("top_"):
            k = key.split("_")[1]
            lines.append(f"Top-{k} Accuracy: {metrics[key]:.4f}")
    
    if "mean_graph_distance" in metrics:
        lines.append(f"Mean Graph Distance Error: {metrics['mean_graph_distance']:.2f}")
    
    lines.append("\n--- Magnitude Estimation ---")
    if "magnitude_mae" in metrics:
        lines.append(f"Magnitude MAE: {metrics['magnitude_mae']:.4f}")
    if "magnitude_rmse" in metrics:
        lines.append(f"Magnitude RMSE: {metrics['magnitude_rmse']:.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
