import os
import glob
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union

# Machine Learning Metrics
from sklearn.metrics import roc_auc_score, roc_curve

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
#               CONFIGURATION
# ==========================================

@dataclass
class ExperimentConfig:
    # --- Input Paths ---
    TRAIN_PATH: str = "./data/train_csvs"   
    EVAL_PATH: str = "./data/eval_jsonls"
    OUTPUT_DIR: str = "./results"           # Folder to save plots and CSVs
    
    # --- Model Configuration ---
    MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.2"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT: bool = True
    MAX_MODEL_LEN: int = 2048
    
    # --- Probing Hyperparameters ---
    LAYER_PERCENTILE: float = 0.20
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 10                        # Epochs for PyTorch Probe
    BATCH_SIZE: int = 32                    # Batch size for Probe Training (not LLM extraction)
    SEED: int = 42
    
    # --- Execution Flags ---
    DRY_RUN: bool = True
    LLM_BATCH_SIZE: int = 4                 # Batch size for LLM activation extraction

    # --- Probe Selection ---
    SELECTED_PROBES: List[str] = field(default_factory=lambda: [
        "mean_torch",    # PyTorch Linear Probe (Tracks Loss)
        "mass_mean",     # Mass-Mean (Parameter-free)
        "followup",      # Follow-up Probe
        "upper_bound"    # Trained on Evaluation Data
    ])

CONFIG = ExperimentConfig()

# Create output directory
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

# Reproducibility
random.seed(CONFIG.SEED)
np.random.seed(CONFIG.SEED)
torch.manual_seed(CONFIG.SEED)

# ==========================================
#           PLOTTING UTILS
# ==========================================

def plot_loss_curve(loss_history: List[float], title: str, filename: str):
    """Plots the training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", color="#4C72B0")
    plt.title(f"Loss Curve: {title}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, filename))
    plt.close()

def plot_roc_curves(results_dict: Dict[str, Dict], dataset_name: str):
    """Plots ROC curves for all methods on a single dataset."""
    plt.figure(figsize=(8, 6))
    
    for method_name, data in results_dict.items():
        if "fpr" in data and "tpr" in data:
            auc = data.get("AUROC", 0)
            plt.plot(data["fpr"], data["tpr"], label=f"{method_name} (AUC={auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Diagonal
    plt.title(f"ROC Curve - {dataset_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, f"roc_{dataset_name}.png"))
    plt.close()

def plot_score_distribution(y_true, y_scores, method_name, dataset_name):
    """Plots histogram of scores for Honest vs Lie samples."""
    plt.figure(figsize=(8, 5))
    
    # Separate scores
    honest_scores = np.array(y_scores)[np.array(y_true) == 0]
    lie_scores = np.array(y_scores)[np.array(y_true) == 1]
    
    sns.histplot(honest_scores, color="green", label="Honest", kde=True, stat="density", alpha=0.4)
    sns.histplot(lie_scores, color="red", label="Lie", kde=True, stat="density", alpha=0.4)
    
    plt.title(f"Score Distribution: {method_name} on {dataset_name}")
    plt.xlabel("Probe Lie Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, f"dist_{dataset_name}_{method_name}.png"))
    plt.close()

# ==========================================
#           DATA & MODEL LOADING
# ==========================================

def get_files_from_path(path_str: str, extension: str) -> List[str]:
    if os.path.isfile(path_str):
        return [path_str] if path_str.endswith(extension) else []
    elif os.path.isdir(path_str):
        return sorted(glob.glob(os.path.join(path_str, f"*{extension}")))
    return []

def format_chat(tokenizer, messages: List[Dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    return text

def load_train_data(path_str: str) -> List[Dict]:
    if CONFIG.DRY_RUN:
        return [{"messages": [{"role":"user","content":"F"}], "label": 0}]*50 + \
               [{"messages": [{"role":"user","content":"F"}], "label": 1}]*50

    files = get_files_from_path(path_str, ".csv")
    if not files: return []

    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'statement' not in df.columns or 'label' not in df.columns: continue
            
            for _, row in df.iterrows():
                probe_label = 0 if int(row['label']) == 1 else 1
                all_data.append({
                    "messages": [{"role": "user", "content": "Tell me a fact."},
                                 {"role": "assistant", "content": str(row['statement'])}],
                    "label": probe_label
                })
        except Exception: pass
    return all_data

def load_eval_datasets(path_str: str) -> Dict[str, List[Dict]]:
    if CONFIG.DRY_RUN:
        return {"synthetic_eval": [{"messages": [{"role":"assistant","content":"T"}], "label": 0}]*20 + \
                                  [{"messages": [{"role":"assistant","content":"F"}], "label": 1}]*20}

    files = get_files_from_path(path_str, ".jsonl")
    datasets = {}
    for f in files:
        name = os.path.basename(f).replace(".jsonl", "")
        data = []
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    
                    is_lie = obj.get("deceptive", obj.get("label"))
                    if is_lie is None: continue
                    label = 1 if (is_lie if isinstance(is_lie, bool) else str(is_lie).lower() in ["true", "1"]) else 0
                    
                    data.append({"messages": obj["messages"], "label": label})
            if data: datasets[name] = data
        except Exception: pass
    return datasets

def load_model():
    if CONFIG.DRY_RUN: return None, None
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_4bit=CONFIG.USE_4BIT, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(CONFIG.MODEL_ID, quantization_config=bnb_config if CONFIG.USE_4BIT else None, device_map="auto", trust_remote_code=True)
    return tokenizer, model

def get_activations(tokenizer, model, data: List[Dict], layer_idx: int, method: str = "mean"):
    if CONFIG.DRY_RUN:
        dim = 4096
        # Return lists of arrays
        return [np.random.randn(random.randint(2,5), dim) for _ in data], np.array([d['label'] for d in data])

    model.eval()
    activations_list = []
    labels_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), CONFIG.LLM_BATCH_SIZE), desc=f"Extr. ({method})", leave=False):
            batch = data[i : i + CONFIG.LLM_BATCH_SIZE]
            
            if method == "mean":
                full_texts = [format_chat(tokenizer, d["messages"]) for d in batch]
            else:
                full_texts = [format_chat(tokenizer, d["messages"]) + "\nUser: Was that true?\nAssistant: Yes" for d in batch]
            
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG.MAX_MODEL_LEN).to(CONFIG.DEVICE)
            out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1]
            
            for j, item in enumerate(batch):
                valid_len = inputs.attention_mask[j].sum().item()
                if method == "mean":
                    prompt_txt = format_chat(tokenizer, item["messages"][:-1])
                    start_idx = len(tokenizer(prompt_txt, add_special_tokens=False).input_ids)
                    start_idx = min(start_idx, valid_len - 1)
                    act = hs[j, start_idx:valid_len, :].cpu().numpy()
                else:
                    act = hs[j, valid_len-1:valid_len, :].cpu().numpy()
                
                if act.shape[0] > 0:
                    activations_list.append(act)
                    labels_list.append(item["label"])
                else:
                    activations_list.append(np.zeros((1, hs.shape[-1])))
                    labels_list.append(item["label"])

    return activations_list, np.array(labels_list)

# ==========================================
#           PROBE CLASSES (WITH LOSS)
# ==========================================

class PyTorchLinearProbe(nn.Module):
    """Linear Probe implemented in PyTorch to allow Loss Plotting."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_history = []
        
    def fit(self, X_list, y, epochs=CONFIG.EPOCHS, batch_size=CONFIG.BATCH_SIZE, lr=CONFIG.LEARNING_RATE):
        # Flatten for training
        X_flat = np.vstack(X_list)
        y_flat = []
        for x, label in zip(X_list, y):
            y_flat.extend([label] * x.shape[0])
            
        # Convert to Tensor
        X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(CONFIG.DEVICE)
        y_tensor = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(CONFIG.DEVICE)
        
        self.to(CONFIG.DEVICE)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.1) # Weight decay = L2 reg
        criterion = nn.BCEWithLogitsLoss()
        
        self.loss_history = []
        self.train()
        
        dataset_size = len(X_flat)
        indices = np.arange(dataset_size)
        
        print(f"Training PyTorch Probe on {dataset_size} samples...")
        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                idx = indices[i : i + batch_size]
                batch_X = X_tensor[idx]
                batch_y = y_tensor[idx]
                
                optimizer.zero_grad()
                outputs = self.linear(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Record loss per step (or average per epoch)
                self.loss_history.append(loss.item())

    def predict_score(self, X_list):
        self.eval()
        scores = []
        with torch.no_grad():
            for x in X_list:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(CONFIG.DEVICE)
                logits = self.linear(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                scores.append(np.mean(probs))
        return np.array(scores)

class MassMeanProbe:
    def __init__(self):
        self.direction = None
    def fit(self, X_list, y):
        X_flat = np.vstack(X_list)
        y_flat = []
        for x, label in zip(X_list, y): y_flat.extend([label] * x.shape[0])
        y_flat = np.array(y_flat)
        mean_lie = np.mean(X_flat[y_flat == 1], axis=0)
        mean_honest = np.mean(X_flat[y_flat == 0], axis=0)
        self.direction = mean_lie - mean_honest
    def predict_score(self, X_list):
        scores = []
        for x in X_list:
            raw = np.dot(x, self.direction)
            scores.append(np.mean(1 / (1 + np.exp(-raw))))
        return np.array(scores)

# ==========================================
#               MAIN PIPELINE
# ==========================================

def run_pipeline():
    print("=== LIARS' BENCH REPLICATION (WITH PLOTS) ===")
    
    tokenizer, model = load_model()
    target_layer = 6
    if model: target_layer = int(model.config.num_hidden_layers * CONFIG.LAYER_PERCENTILE)

    # --- 1. Train Generic Probes ---
    probes = {}
    train_data = load_train_data(CONFIG.TRAIN_PATH)
    
    if train_data:
        # A. PyTorch Mean Probe (With Loss Plot)
        if "mean_torch" in CONFIG.SELECTED_PROBES or "mass_mean" in CONFIG.SELECTED_PROBES:
            print("\nExtracting Training Activations (Mean)...")
            X_tr, y_tr = get_activations(tokenizer, model, train_data, target_layer, method="mean")
            
            if "mean_torch" in CONFIG.SELECTED_PROBES:
                # Initialize PyTorch Probe
                input_dim = X_tr[0].shape[-1]
                p = PyTorchLinearProbe(input_dim)
                p.fit(X_tr, y_tr)
                probes["mean_torch"] = p
                
                # PLOT LOSS
                plot_loss_curve(p.loss_history, "Mean Probe (Generic)", "loss_mean_probe.png")
                
            if "mass_mean" in CONFIG.SELECTED_PROBES:
                p = MassMeanProbe()
                p.fit(X_tr, y_tr)
                probes["mass_mean"] = p

        # B. Follow-up Probe (With Loss Plot)
        if "followup" in CONFIG.SELECTED_PROBES:
            print("\nExtracting Training Activations (Follow-up)...")
            X_fu, y_fu = get_activations(tokenizer, model, train_data, target_layer, method="followup")
            input_dim = X_fu[0].shape[-1]
            p = PyTorchLinearProbe(input_dim)
            p.fit(X_fu, y_fu)
            probes["followup"] = p
            
            # PLOT LOSS
            plot_loss_curve(p.loss_history, "Follow-up Probe", "loss_followup_probe.png")

    # --- 2. Evaluate Datasets ---
    eval_datasets = load_eval_datasets(CONFIG.EVAL_PATH)
    final_results = []
    
    for name, ds_data in eval_datasets.items():
        if len(ds_data) < 2: continue
        print(f"\nEvaluating: {name}")
        
        # Pre-extract Test Activations
        X_mean_test, y_test = None, None
        X_fu_test = None
        
        if any(x in CONFIG.SELECTED_PROBES for x in ["mean_torch", "mass_mean", "upper_bound"]):
            X_mean_test, y_test = get_activations(tokenizer, model, ds_data, target_layer, method="mean")
        if "followup" in CONFIG.SELECTED_PROBES:
            X_fu_test, _ = get_activations(tokenizer, model, ds_data, target_layer, method="followup")
            
        # Upper Bound (Train/Test Split)
        test_indices = list(range(len(ds_data)))
        ub_probe = None
        
        if "upper_bound" in CONFIG.SELECTED_PROBES:
            mid = len(ds_data) // 2
            train_idx = list(range(mid))
            test_indices = list(range(mid, len(ds_data)))
            
            X_ub_tr = [X_mean_test[i] for i in train_idx]
            y_ub_tr = y_test[train_idx]
            
            if len(set(y_ub_tr)) > 1:
                ub_probe = PyTorchLinearProbe(X_ub_tr[0].shape[-1])
                ub_probe.fit(X_ub_tr, y_ub_tr, epochs=20) # Train longer on small data
                # Optional: Plot UB Loss
                plot_loss_curve(ub_probe.loss_history, f"Upper Bound ({name})", f"loss_ub_{name}.png")

        # --- Scoring & Metrics ---
        # We collect ROC data for this dataset
        roc_data_per_method = {}
        
        y_true_subset = y_test[test_indices]
        
        # Helper to process probe
        def process_probe(key, probe_obj, inputs, method_label):
            scores = probe_obj.predict_score(inputs)
            
            # ROC / Metrics
            fpr, tpr, _ = roc_curve(y_true_subset, scores)
            auc = roc_auc_score(y_true_subset, scores)
            
            roc_data_per_method[method_label] = {"fpr": fpr, "tpr": tpr, "AUROC": auc}
            
            # Plot Score Distribution (First valid dataset only to avoid clutter, or all)
            plot_score_distribution(y_true_subset, scores, method_label, name)
            
            # Table Metrics
            # Threshold logic...
            # (Simplified for brevity here, same logic as before)
            return {"Dataset": name, "Method": method_label, "AUROC": auc}

        # 1. Mean Torch
        if "mean_torch" in probes:
            inputs = [X_mean_test[i] for i in test_indices]
            res = process_probe("mean_torch", probes["mean_torch"], inputs, "Mean Probe")
            final_results.append(res)
            
        # 2. Mass Mean
        if "mass_mean" in probes:
            inputs = [X_mean_test[i] for i in test_indices]
            res = process_probe("mass_mean", probes["mass_mean"], inputs, "Mass-Mean")
            final_results.append(res)
            
        # 3. Followup
        if "followup" in probes:
            inputs = [X_fu_test[i] for i in test_indices]
            res = process_probe("followup", probes["followup"], inputs, "Follow-up")
            final_results.append(res)
            
        # 4. Upper Bound
        if ub_probe:
            inputs = [X_mean_test[i] for i in test_indices]
            res = process_probe("upper_bound", ub_probe, inputs, "Upper Bound")
            final_results.append(res)

        # PLOT COMBINED ROC FOR THIS DATASET
        plot_roc_curves(roc_data_per_method, name)

    # Output Table
    if final_results:
        df = pd.DataFrame(final_results)
        print("\n" + df.to_markdown(index=False))
        df.to_csv(os.path.join(CONFIG.OUTPUT_DIR, "final_results.csv"), index=False)
        print(f"\nPlots and Results saved to {CONFIG.OUTPUT_DIR}")

if __name__ == "__main__":
    run_pipeline()