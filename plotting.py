print("Importing matplotlib.pyplot")
import matplotlib.pyplot as plt
print("Importing seaborn")
import seaborn as sns
print("Importing os")
import os
print("Importing numpy")
import numpy as np
from typing import List, Dict, Tuple

from probe_config import CONFIG
import json
import glob

# ==========================================
#             PLOTTING UTILS
# ==========================================

def plot_loss_curve(loss_history: List[float], title: str, filename: str):
    """Plots training loss to verify convergence."""
    if not loss_history: return
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="BCE Loss (with L2 Reg)", color="#4C72B0", linewidth=2)
    plt.title(f"Training Convergence: {title}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, filename))
    plt.close()

def plot_roc_curves(roc_data_dict: Dict[str, Dict], dataset_name: str, model_name: str):
    """
    Plots ROC curves for all probes on a single dataset.
    """
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    sorted_items = sorted(roc_data_dict.items(), key=lambda x: x[1].get("auc", 0), reverse=True)

    for i, (probe_name, data) in enumerate(sorted_items):
        if "fpr" in data and "tpr" in data:
            auc = data.get("auc", 0)
            style = '--' if "Upper Bound" in probe_name else '-'
            width = 2.5 if "Upper Bound" in probe_name else 2.0
            plt.plot(data["fpr"], data["tpr"],
                     label=f"{probe_name} (AUC={auc:.2f})",
                     color=colors[i % len(colors)], linestyle=style, linewidth=width)

    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label="Random Chance")
    plt.title(f"ROC Curve - {dataset_name} ({model_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_model = model_name.replace("/", "_")

    eval_dir = CONFIG.OUT_EVAL_DIR.format(model_name=model_name, dataset_name=dataset_name)
    os.makedirs(eval_dir, exist_ok=True)
    plt.savefig(os.path.join(eval_dir, f"roc.png"))

    plt.close()

def plot_score_distribution(y_true, y_scores, method_name, dataset_name, model_name):
    """Visualizes separation between Honest (0) and Lie (1) scores."""
    plt.figure(figsize=(8, 5))
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    honest_scores = y_scores[y_true == 0]
    lie_scores = y_scores[y_true == 1]

    sns.histplot(honest_scores, color="green", label="Honest", kde=True, stat="probability", alpha=0.3, element="step")
    sns.histplot(lie_scores, color="red", label="Lie", kde=True, stat="probability", alpha=0.3, element="step")

    plt.title(f"Score Dist: {method_name} on {dataset_name}")
    plt.xlabel("Probe Lie Probability")
    plt.legend()
    plt.tight_layout()

    safe_model = model_name.replace("/", "_")
    safe_method = method_name.replace(" ", "_")
    
    eval_dir = CONFIG.OUT_EVAL_DIR.format(model_name=safe_model, dataset_name=dataset_name)
    os.makedirs(eval_dir, exist_ok=True)
    plt.savefig(os.path.join(eval_dir, f"dist_{safe_method}.png"))

    plt.close()
