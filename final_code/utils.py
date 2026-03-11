"""
utils.py

Contains helper functions for logging, metric calculation, plotting, 
and safe (atomic) file checkpointing.
"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve

# Import global config
from config import CONFIG

def print_info(msg: str):
    """Prints a timestamped message to the console and flushes the buffer (crucial for SLURM logs)."""
    if CONFIG.VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def show_device_info():
    """Logs the available GPU resources and what device PyTorch is using."""
    gpu_avail = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    print_info(f"CUDA available: {gpu_avail}, GPU count: {n_gpus}, CONFIG.DEVICE: {CONFIG.DEVICE}")
    if gpu_avail:
        for i in range(n_gpus):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = "Unknown"
            print_info(f"  GPU {i}: {name}")

# ==========================================
# Plotting & Metrics
# ==========================================

def plot_loss_curve(loss_history: List[float], title: str, filename: str):
    """Saves a plot of the training loss over epochs/steps."""
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
    Plots ROC curves for all evaluated probes on a single dataset to compare performance.
    Saves the figure to the configured output directory.
    """
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # Sort items by AUC descending so the best probes are listed first in the legend
    sorted_items = sorted(roc_data_dict.items(), key=lambda x: x[1].get("auc", 0), reverse=True)
    
    for i, (probe_name, data) in enumerate(sorted_items):
        if "fpr" in data and "tpr" in data:
            auc = data.get("auc", 0)
            # Make the Upper Bound curve visually distinct (dashed and thicker)
            style = '--' if "upper" in probe_name.lower() else '-'
            width = 2.5 if "upper" in probe_name.lower() else 2.0
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
    
    # Clean up strings for safe file naming
    safe_model = model_name.replace("/", "_")
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, f"roc_{safe_model}_{dataset_name}.png"))
    plt.close()

def plot_score_distribution(y_true, y_scores, method_name, dataset_name, model_name):
    """Plots the overlapping histograms of predicted scores for Honest vs Lie samples."""
    plt.figure(figsize=(8, 5))
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    honest_scores = y_scores[y_true == 0]
    lie_scores = y_scores[y_true == 1]
    
    sns.histplot(honest_scores, label="Honest", color="green", kde=True, stat="density", alpha=0.3, element="step")
    sns.histplot(lie_scores, label="Lie", color="red", kde=True, stat="density", alpha=0.3, element="step")
    plt.title(f"Score Dist: {method_name} on {dataset_name}")
    plt.xlabel("Probe Lie Probability")
    plt.legend()
    plt.tight_layout()
    
    safe_model = model_name.replace("/", "_")
    safe_method = method_name.replace(" ", "_")
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, f"dist_{safe_model}_{dataset_name}_{safe_method}.png"))
    plt.close()

def calculate_extended_metrics(y_true, y_scores):
    """
    Calculates AUROC, Recall, and Balanced Accuracy at specific strict False Positive Rates.
    Returns a dictionary of metrics and a dictionary of plot data.
    """
    if len(np.unique(y_true)) < 2: return {}, {}
    fpr_array, tpr_array, thresholds = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    plot_data = {"fpr": fpr_array, "tpr": tpr_array, "auc": auroc}
    metrics = {"AUROC": auroc}
    
    # Calculate performance at strict specific thresholds
    targets = [0.01, 0.001, 0.0001]
    for target_fpr in targets:
        valid_indices = np.where(fpr_array <= target_fpr)[0]
        idx = valid_indices[-1] if len(valid_indices) > 0 else 0
        actual_fpr = fpr_array[idx]
        actual_tpr = tpr_array[idx]
        actual_tnr = 1.0 - actual_fpr
        bal_acc = (actual_tpr + actual_tnr) / 2.0
        
        suffix = f"{int(target_fpr*100)}pctFPR" if target_fpr >= 0.001 else f"{target_fpr}"
        metrics[f"BALANCEDACCURACYAT{suffix}"] = bal_acc
        metrics[f"RECALLAT{suffix}"] = actual_tpr
        metrics[f"FPRAT{suffix}"] = actual_fpr
        
    y_pred = (y_scores > 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    metrics["RECALL"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return metrics, plot_data

# ==========================================
# File & Checkpoint Utilities
# ==========================================

def _ensure_dir(path: str):
    try: os.makedirs(path, exist_ok=True)
    except Exception: pass

def _atomic_save_torch(obj, path: str):
    """
    Saves a torch object to a temporary file first, then atomically replaces the target file.
    Prevents corrupting checkpoints if the process is killed (e.g., by SLURM time limits).
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _save_scaler_object(obj, path: str):
    """Saves the scikit-learn StandardScaler atomically using joblib or pickle."""
    tmp = path + ".tmp"
    try:
        import joblib as _joblib
        _joblib.dump(obj, tmp)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            import pickle
            with open(tmp, "wb") as fh:
                pickle.dump(obj, fh)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print_info(f"[Checkpoint] failed writing scaler: {e}")
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            return False

def checkpoint_basename(prefix: str, model_name: Optional[str] = None, dataset_name: Optional[str] = None, epoch: Optional[int] = None):
    """Generates a system-safe filename for checkpoints without the extension."""
    parts = []
    if prefix: parts.append(prefix)
    if model_name: parts.append(model_name.replace("/", "_"))
    if dataset_name: parts.append(dataset_name.replace("/", "_"))
    if epoch is not None: parts.append(f"epoch{int(epoch)}")
    base = "__".join([p for p in parts if p])
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)

def rotate_checkpoints_dir(checkpoint_dir: str, base_prefix: str, keep_last_k: Optional[int]):
    """
    Deletes older checkpoints in a directory to prevent disk overflow.
    Keeps only the `keep_last_k` most recently modified files matching the prefix.
    """
    try:
        if not keep_last_k or keep_last_k <= 0: return
        _ensure_dir(checkpoint_dir)
        
        candidates = []
        for fn in os.listdir(checkpoint_dir):
            if base_prefix in fn and (fn.endswith(".pth") or fn.endswith(".pt") or fn.endswith(".pth.tmp")):
                fp = os.path.join(checkpoint_dir, fn)
                try:
                    mtime = os.path.getmtime(fp)
                    candidates.append((mtime, fp))
                except Exception: pass
                
        # Sort by newest first, then slice to find the ones to delete
        candidates.sort(reverse=True, key=lambda x: x[0])
        to_delete = candidates[keep_last_k:]
        
        for _, fp in to_delete:
            try:
                os.remove(fp)
                print_info(f"[CheckpointRotate] removed old checkpoint {fp}")
            except Exception as e:
                print_info(f"[CheckpointRotate] failed to remove {fp}: {e}")
    except Exception as e:
        print_info(f"[CheckpointRotate] unexpected error: {e}")