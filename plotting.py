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
    if not loss_history:
        return
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

    sorted_items = sorted(
        roc_data_dict.items(), key=lambda x: x[1].get("auc", 0), reverse=True
    )

    for i, (probe_name, data) in enumerate(sorted_items):
        if "fpr" in data and "tpr" in data:
            auc = data.get("auc", 0)
            style = "--" if "Upper Bound" in probe_name else "-"
            width = 2.5 if "Upper Bound" in probe_name else 2.0
            plt.plot(
                data["fpr"],
                data["tpr"],
                label=f"{probe_name} (AUC={auc:.2f})",
                color=colors[i % len(colors)],
                linestyle=style,
                linewidth=width,
            )

    plt.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Chance")
    plt.title(f"ROC Curve - {dataset_name} ({model_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_model = model_name.replace("/", "_")

    eval_dir = CONFIG.OUT_EVAL_DIR.format(
        model_name=model_name, dataset_name=dataset_name
    )
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

    sns.histplot(
        honest_scores,
        color="green",
        label="Honest",
        kde=True,
        stat="probability",
        alpha=0.3,
        element="step",
    )
    sns.histplot(
        lie_scores,
        color="red",
        label="Lie",
        kde=True,
        stat="probability",
        alpha=0.3,
        element="step",
    )

    plt.title(f"Score Dist: {method_name} on {dataset_name}")
    plt.xlabel("Probe Lie Probability")
    plt.legend()
    plt.tight_layout()

    safe_model = model_name.replace("/", "_")
    safe_method = method_name.replace(" ", "_")

    eval_dir = CONFIG.OUT_EVAL_DIR.format(
        model_name=safe_model, dataset_name=dataset_name
    )
    os.makedirs(eval_dir, exist_ok=True)
    plt.savefig(os.path.join(eval_dir, f"dist_{safe_method}.png"))

    plt.close()


def plot_metric_comparison():
    """Plot AUROC comparison across models and datasets."""
    
    # Data from the table
    models = [
        "Llama 3.3 70B", 
        # "Qwen 2.5 72B", 
        # "Mistral 3.1 24B", 
        # "Gemma 3 27B"
    ]
    
    # AUROC scores (None for missing values)
    datasets = ["HP-C", "HP-KR", "CG", "ST", "ID", "GS", "IT"]
    liars_bench_auroc = {
        "Llama 3.3 70B": [0.41, 0.12, 1.00, 0.57, 0.95, 0.45, 0.93],      
        "Qwen 2.5 72B": [0.50, 0.49, None, 0.57, 0.60, None, None],
        "Mistral 3.1 24B": [0.55, 0.46, None, 0.82, 0.69, 0.58, 0.58],
        "Gemma 3 27B": [None, None, 0.70, None, 0.58, None, 0.30],
    }

    # Column-order: CG, GS, HP-C, HP-KR
    column_order = [2, 5, 0, 1]
    our_auroc = {
        "Llama 3.3 70B": [0.724907063197026, 0.4864219270998931, 0.4523625291528656, 0.3736713008003614],
        # "Llama 3.3 70B": [0.8002973977695167, 0.4793944285469709,0.6041388362354713, 0.3449852712363687],
        # "Mistral 3.1 24B": [0.5697902283590015, 0.5166841946502964, 0.5031429522369988, None ],
    }
    # Reorder data according to column_order
    liars_bench_auroc = {
        model: [liars_bench_auroc[model][i] for i in column_order]
        for model in models
    }
    datasets = [datasets[i] for i in column_order]
    print("Columns:", len(liars_bench_auroc[models[0]]))
    print("Our_Columns:", len(our_auroc[models[0]]))

    # Side-by-side bars per model, per dataset: data vs our_data
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(datasets))
    model_width = 0.8 / max(1, len(models))
    bar_width = model_width / 2

    # Color scheme: each model gets a base color, data is darker, our_data is lighter
    model_colors = {
        "Llama 3.3 70B": {"our_data": "#FF8C00", "data": "#FFD699"},  # Dark orange, pastel orange
        "Mistral 3.1 24B": {"our_data": "#1f77b4", "data": "#A9C5E0"},  # Dark blue, pastel blue
    }

    for m_idx, model in enumerate(models):
        model_offset = (m_idx - (len(models) - 1) / 2) * model_width

        for d_idx, dataset in enumerate(datasets):
            base = liars_bench_auroc[model][d_idx]
            ours = our_auroc[model][d_idx]

            base_height = base if base is not None else 0
            ours_height = ours if ours is not None else 0

            base_bar = ax.bar(
                x[d_idx] + model_offset - bar_width / 2,
                base_height,
                bar_width,
                color=model_colors[model]["data"],
                alpha=0.85,
            )
            if base is None:
                base_bar[0].set_hatch("///")
                base_bar[0].set_alpha(0.3)

            ours_bar = ax.bar(
                x[d_idx] + model_offset + bar_width / 2,
                ours_height,
                bar_width,
                color=model_colors[model]["our_data"],
                alpha=0.85,
            )
            if ours is None:
                ours_bar[0].set_hatch("///")
                ours_bar[0].set_alpha(0.3)

    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="random-choice model")

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel("AUROC Score", fontsize=12, fontweight="bold")
    ax.set_title("Liars Bench vs our replication per Model and Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Legend for data vs our_data with per-model colors
    handles = []
    labels = []
    for model in models:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=model_colors[model]["data"], alpha=0.85))
        labels.append(f"{model} (Liars-Bench)")
        handles.append(plt.Rectangle((0, 0), 1, 1, color=model_colors[model]["our_data"], alpha=0.85))
        labels.append(f"{model} (our replication)")
    
    ax.legend(handles, labels, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(CONFIG.OUTPUT_DIR, "data_vs_our_data_grouped_bar.png"), dpi=300
    )
    plt.close()
    
    

if __name__ == "__main__":
    plot_metric_comparison()
