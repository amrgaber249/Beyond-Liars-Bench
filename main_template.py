# ---------- Imports ----------
print("Importing os, json, random, numpy")
import os, json, random, numpy as np

print("Importing pandas, matplotlib")
import pandas as pd, matplotlib.pyplot as plt

print("Importing torch")
import torch

print("Importing transformers")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Importing datasets (Hugging Face)")
from datasets import load_dataset, get_dataset_config_names
print("Defining lightweight dialogue types")
from dataclasses import dataclass
from typing import List

@dataclass
class Message:
    role: str
    content: str

Dialogue = List[Message]


print("Importing sklearn components")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    recall_score,
    confusion_matrix,
)


# ---------- CONFIG ----------
# Use the original model and quantize on load with bitsandbytes
MODEL_ID="distilgpt2"  # smaller model for testing
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
DRY_RUN = False  # True -> quick synthetic features; False -> attempt to load MODEL_ID
# We will quantize with bitsandbytes instead of loading a pre-quantized checkpoint
QUANT_TYPE = "4bit"  # "" for no quantization, "4bit" or "8bit"
MAX_MODEL_LEN = 1024  # Modest context to reduce VRAM
# Available configs for "Cadenza-Labs/liars-bench":
# ['alpaca', 'convincing-game', 'gender-secret', 'harm-pressure-choice', 'harm-pressure-knowledge-report', 'insider-trading', 'instructed-deception', 'soft-trigger']
SELECT_DATASETS = ['alpaca', 'convincing-game', 'gender-secret', 'harm-pressure-choice', 'harm-pressure-knowledge-report', 'insider-trading', 'instructed-deception', 'soft-trigger'] 
MAX_EXAMPLES_PER_DATASET = 300
LAYER_TO_EXTRACT = -1
PROBE_C = 1.0
TEST_SIZE = 0.2
RANDOM_SEED = 42
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Upper-bound options:
# 'merged_inclusive': train probe on union of all datasets' training splits (privileged labels available)
# 'leave_one_out'    : for each dataset, train on union of *other* datasets' training splits
UPPER_BOUND_MODE = "merged_inclusive"  # or "leave_one_out"
TRAIN_UPPER_BOUND = True  # whether to run the upper-bound experiments

# set seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------- dataset loader ----------
def load_datasets(keys, max_per_dataset=200):
    datasets_out = {}
    hf_dataset_id = "Cadenza-Labs/liars-bench"
    try:
        hf_configs = get_dataset_config_names(hf_dataset_id)
        print("Available HF configs:", hf_configs)
    except Exception as e:
        print("Could not list HF configs:", e)
        hf_configs = []

    for k in keys:
        if os.path.exists(k):
            # load local jsonl
            arr = []
            try:
                with open(k, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= max_per_dataset:
                            break
                        try:
                            obj = json.loads(line)
                            label = (
                                obj.get("deceptive")
                                if "deceptive" in obj
                                else obj.get("label")
                                if "label" in obj
                                else obj.get("is_lie")
                                if "is_lie" in obj
                                else obj.get("honest")
                                if "honest" in obj
                                else obj.get("y")
                            )
                            if label is None:
                                label = 0
                            if isinstance(label, bool):
                                label = int(label)
                            try:
                                label = int(label)
                            except Exception:
                                label = 1 if str(label).lower() in ("1","true","lie","falsehood") else 0
                            messages = obj.get("messages")
                            if messages and isinstance(messages, list):
                                norm = []
                                for m in messages:
                                    content = str(m.get("content", "")).replace("\xa0", " ").rstrip("<>").strip()
                                    norm.append({"role": m.get("role", "user"), "content": content})
                                arr.append({"messages": norm, "label": label})
                            else:
                                text = (
                                    obj.get("prompt")
                                    or obj.get("text")
                                    or obj.get("context")
                                    or obj.get("input")
                                    or str(obj)
                                )
                                arr.append({"text": text, "label": int(label)})
                        except Exception:
                            continue
                print(f"Loaded {len(arr)} examples from file {k}")
                datasets_out[k] = arr
                continue
            except Exception as e:
                print(
                    f"Failed reading file {k}, will try HF dataset or synthetic. Error:",
                    e,
                )

        # Try Hugging Face dataset config matching key `k`
        if k in hf_configs:
            try:
                ds = load_dataset(hf_dataset_id, k)
                arr = []
                total = 0
                for split_name, split_ds in ds.items():
                    for row in split_ds:
                        # Extract messages and label robustly
                        messages = row.get("messages")
                        label = (
                            row.get("deceptive")
                            if "deceptive" in row
                            else row.get("label")
                            if "label" in row
                            else row.get("honest")
                            if "honest" in row
                            else row.get("y")
                        )
                        if label is None:
                            # default to 0
                            label = 0
                        if isinstance(label, bool):
                            label = int(label)
                        try:
                            label = int(label)
                        except Exception:
                            # map non-integer labels if present
                            label = (
                                1
                                if str(label).lower()
                                in ("1", "true", "lie", "falsehood")
                                else 0
                            )
                        if messages and isinstance(messages, list):
                            norm = []
                            for m in messages:
                                cleaned = str(m.get("content", "")).replace("\xa0", " ").rstrip("<>").strip()
                                norm.append({"role": m.get("role", "user"), "content": cleaned})
                            arr.append({"messages": norm, "label": label})
                        else:
                            text = (
                                row.get("prompt")
                                or row.get("text")
                                or row.get("context")
                                or row.get("input")
                                or row.get("question")
                                or row.get("response")
                                or str(row)
                            )
                            arr.append({"text": text, "label": label})
                        total += 1
                        if total >= max_per_dataset:
                            break
                    if total >= max_per_dataset:
                        break
                print(f"Loaded {len(arr)} examples from HF config {k}")
                datasets_out[k] = arr
                continue
            except Exception as e:
                print(f"Failed to load HF dataset config {k}:", e)

        # Fallback: synthetic alternating examples
        arr = []
        for i in range(max_per_dataset):
            lbl = i % 2
            arr.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"SYN-{k}-{i}: Provide a statement."},
                    {"role": "assistant", "content": f"This is {'a lie' if lbl==1 else 'an honest'} response."}
                ],
                "label": lbl
            })
        print(f"Created synthetic dataset {k} with {len(arr)} examples (fallback)")
        datasets_out[k] = arr

    return datasets_out

def render_dialogue(messages: Dialogue, tokenizer: AutoTokenizer) -> str:
    # Use tokenizer chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    # Fallback: simple join
    return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])


# ---------- Model load & pad_token fix ----------
def safe_load_model(model_id, device="cpu", try_fp16=False, max_model_len=512):
    """
    Attempts to load a causal LM using transformers (optionally with bitsandbytes 4-bit).
    Returns tokenizer, model (or (None, None) on failure).
    """
    try:
        # load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

        # If tokenizer has no pad_token, try to set it to eos_token; else add '[PAD]'
        added_special = False
        if tokenizer.pad_token is None:
            if getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Tokenizer had no pad_token → set pad_token = eos_token")
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                added_special = True
                print(
                    "Tokenizer had no pad_token or eos_token → added [PAD] as pad_token"
                )

        # Load with transformers + bitsandbytes 4-bit quantization
        print(
            "Loading model with transformers + bitsandbytes 4-bit (device_map=auto)..."
        )
        max_mem = None
        if device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                per_gpu_mem = os.environ.get("GPU_MEM_PER_DEV", "46GiB")
                max_mem = {i: per_gpu_mem for i in range(gpu_count)}
        kwargs = {
            "output_hidden_states": True,
            "trust_remote_code": True,
        }
        if QUANT_TYPE in ("4bit", "8bit"):
            if QUANT_TYPE == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if try_fp16 else torch.float16,
            )
            elif QUANT_TYPE == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.bfloat16 if try_fp16 else torch.float16,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4",
                )
            
        if max_mem:
            kwargs["max_memory"] = max_mem
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        # incase we added a special token to tokenizer, resize embeddings
        if added_special:
            model.resize_token_embeddings(len(tokenizer))
        # also ensure tokenizer.padding_side exists
        if not hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        print(f"Loaded model/tokenizer: {model_id}")
        return tokenizer, model
    except Exception as e:
        print("Model load failed:", e)
        return None, None

        # ---------- Activation extraction ----------


def get_mean_activations(
    tokenizer, model, texts, layer_index=-1, device="cpu", max_length=512, batch_size=1
):
    """
    Returns shape (n_texts, hidden_size) of mean-pooled hidden states at layer_index.
    If model/tokenizer None -> raises.
    """

    model.eval()
    dev = torch.device(
        device if (torch.cuda.is_available() and device == "cuda") else "cpu"
    )
    # Detect device mapping and whether model spans multiple devices
    is_sharded = False
    single_model_device = None
    if (
        hasattr(model, "hf_device_map")
        and isinstance(getattr(model, "hf_device_map"), dict)
        and len(model.hf_device_map) > 0
    ):
        devices = set(model.hf_device_map.values())
        # If all modules map to a single device (e.g., 'cuda:3'), treat as single-device
        if len(devices) == 1:
            single_model_device = next(iter(devices))
        else:
            is_sharded = True
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # If items are dialogues, render to strings via chat template
            formatted = []
            for item in batch:
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict) and "role" in item[0] and "content" in item[0]:
                    formatted.append(render_dialogue(item, tokenizer))
                elif isinstance(item, dict) and "messages" in item:
                    formatted.append(render_dialogue(item["messages"], tokenizer))
                else:
                    formatted.append(str(item))
            enc = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            # For sharded models, keep inputs on CPU and let HF Accelerate scatter.
            # If model is on a single device (including via hf_device_map), move inputs there.
            if is_sharded:
                pass  # keep CPU
            elif single_model_device is not None:
                try:
                    enc = enc.to(torch.device(single_model_device))
                except Exception:
                    enc = enc.to(dev)
            else:
                enc = enc.to(dev)
            # forward
            out = model(**enc, output_hidden_states=True)
            # prefer (hidden_states); fallback to last_hidden_state if needed
            if hasattr(out, "hidden_states") and out.hidden_states is not None:
                hs = out.hidden_states[layer_index]  # (bs, seq_len, hidden)
            elif hasattr(out, "last_hidden_state"):
                hs = out.last_hidden_state
            else:
                raise RuntimeError(
                    "Model output has no hidden states or last_hidden_state."
                )
            # Ensure mask and hidden state are on the same device
            mask = enc.attention_mask.unsqueeze(-1).to(hs.device)
            masked = hs * mask
            summed = masked.sum(dim=1)
            lens = mask.sum(dim=1).clamp(min=1)
            meanpooled = summed / lens
            all_vecs.append(meanpooled.cpu().numpy())
    return np.vstack(all_vecs)


# ---------- Probe training & metrics ----------
def train_probe(X_train, y_train, C=1.0, random_state=42):
    clf = LogisticRegression(
        C=C, solver="liblinear", max_iter=2000, random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf


def eval_probe(clf, X_test, y_test, thresh=0.5):
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= thresh).astype(int)
    auroc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    bacc = balanced_accuracy_score(y_test, preds)
    recall_pos = recall_score(y_test, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "auroc": float(auroc),
        "balanced_acc": float(bacc),
        "recall_pos": float(recall_pos),
        "fpr": float(fpr),
    }


# ---------- Experiment runner with upper-bound variants ----------
def run_experiments(
    datasets,
    model_id=None,
    dry_run=True,
    device="cpu",
    layer_index=-1,
    probe_C=1.0,
    test_size=0.2,
    random_seed=42,
    train_upper_bound=True,
    upper_bound_mode="merged_inclusive",
):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tokenizer = model = None
    # attempt to load model if not dry-run
    if not dry_run and model_id:
        print("Attempting to load model:", model_id)
        max_len = globals().get("MAX_MODEL_LEN", 512)  # Get from config
        tokenizer, model = safe_load_model(
            model_id, device=device, max_model_len=max_len
        )
        if tokenizer is None:
            print("Model unavailable; will fallback to synthetic numeric features.")
    results = {}
    # Per-dataset: split into train/test, then train probe on dataset train and test on dataset test
    splits = {}
    for key, examples in datasets.items():
        # Use dialogues when available; else plain text
        texts = [ex.get("messages") if "messages" in ex else ex["text"] for ex in examples]
        labels = np.array([int(ex["label"]) for ex in examples])
        if len(set(labels)) < 2:
            print(f"Skipping {key}: only one label present.")
            continue
        # split
        X_indices = np.arange(len(texts))
        # maintain stratify if possible
        strat = labels if len(set(labels)) > 1 else None
        train_idx, test_idx = train_test_split(
            X_indices, test_size=test_size, random_state=random_seed, stratify=strat
        )
        splits[key] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "texts": texts,
            "labels": labels,
        }

    # Pre-extract activations per dataset (or fallback synthetic)
    activations = {}
    for key, info in splits.items():
        try:
            if tokenizer is None or model is None:
                raise RuntimeError("Model/tokenizer unavailable (dry-run).")
            # extract all activations for this dataset
            A = get_mean_activations(
                tokenizer, model, info["texts"], layer_index=layer_index, device=device
            )
            print(f"Extracted activations for {key}: {A.shape}")
        except Exception as e:
            print(f"Activation extraction failed for {key}:", e)
            print(" -> Falling back to synthetic random features for this dataset.")
            A = np.random.normal(size=(len(info["texts"]), 768))
        activations[key] = A

    # Train per-dataset probes (realistic mean-probe)
    for key, info in splits.items():
        A = activations[key]
        y = info["labels"]
        train_i = info["train_idx"]
        test_i = info["test_idx"]
        X_tr = A[train_i]
        X_te = A[test_i]
        y_tr = y[train_i]
        y_te = y[test_i]
        clf = train_probe(X_tr, y_tr, C=probe_C, random_state=random_seed)
        metrics = eval_probe(clf, X_te, y_te)
        results[key] = metrics
        print(f"[MEAN PROBE] {key}: {metrics}")

    # UPPER-BOUND: privileged-data probe
    if train_upper_bound and len(splits) > 0:
        if upper_bound_mode == "merged_inclusive":
            # Build merged training set by concatenating all datasets' TRAIN splits (privileged data)
            X_tr_list = []
            y_tr_list = []
            for key, info in splits.items():
                A = activations[key]
                y = info["labels"]
                tr = info["train_idx"]
                X_tr_list.append(A[tr])
                y_tr_list.append(y[tr])
            X_train_merged = np.vstack(X_tr_list)
            y_train_merged = np.concatenate(y_tr_list)
            print(
                "Upper-bound (merged_inclusive): merged training size:",
                X_train_merged.shape[0],
            )
            # Train one probe on merged training, evaluate on each dataset's TEST split
            clf_ub = train_probe(
                X_train_merged, y_train_merged, C=probe_C, random_state=random_seed
            )
            for key, info in splits.items():
                A = activations[key]
                y = info["labels"]
                test_i = info["test_idx"]
                metrics = eval_probe(clf_ub, A[test_i], y[test_i])
                results[f"UPPER_MERGED->{key}"] = metrics
                print(f"[UPPER_MERGED] test on {key}: {metrics}")

        elif upper_bound_mode == "leave_one_out":
            # for each dataset: train on union of other datasets' TRAIN splits and test on that dataset's TEST split
            for target_key in splits.keys():
                X_tr_list = []
                y_tr_list = []
                for key, info in splits.items():
                    if key == target_key:
                        continue
                    A = activations[key]
                    y = info["labels"]
                    tr = info["train_idx"]
                    X_tr_list.append(A[tr])
                    y_tr_list.append(y[tr])
                if len(X_tr_list) == 0:
                    print(
                        "No other datasets to train on for leave_one_out; skipping",
                        target_key,
                    )
                    continue
                X_train_loo = np.vstack(X_tr_list)
                y_train_loo = np.concatenate(y_tr_list)
                clf_ub = train_probe(
                    X_train_loo, y_train_loo, C=probe_C, random_state=random_seed
                )
                # test on target's test split
                A_target = activations[target_key]
                y_target = splits[target_key]["labels"]
                test_i = splits[target_key]["test_idx"]
                metrics = eval_probe(clf_ub, A_target[test_i], y_target[test_i])
                results[f"UPPER_LOO->{target_key}"] = metrics
                print(f"[UPPER_LOO] test on {target_key}: {metrics}")
        else:
            print("Unknown upper_bound_mode:", upper_bound_mode)
    return results


# ---------- Reporting ----------
def report_results(results):
    if len(results) == 0:
        print("No results.")
        return
    df = pd.DataFrame.from_dict(results, orient="index")
    df.loc["MEAN"] = df.mean(axis=0)
    print("\nResults table (with MEAN row):")
    print(df)
    metrics = ["auroc", "balanced_acc", "recall_pos", "fpr"]
    keys = list(results.keys())
    arr = np.array([[results[k][m] for m in metrics] for k in keys])
    x = np.arange(len(keys))
    width = 0.18
    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, arr[:, i], width, label=m)
    plt.xticks(x, keys, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Probe metrics by dataset (FPR lower better; others higher better)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    print("=== AI Deception Detection Probe Experiment ===")
    print(f"Model: {MODEL_ID}, Dry-run: {DRY_RUN}, Device: {DEVICE}")
    print(f"Datasets: {SELECT_DATASETS}, Max per dataset: {MAX_EXAMPLES_PER_DATASET}")
    print(f"Layer: {LAYER_TO_EXTRACT}, Probe C: {PROBE_C}, Test size: {TEST_SIZE}")
    print(
        f"Upper-bound mode: {UPPER_BOUND_MODE}, Train upper-bound: {TRAIN_UPPER_BOUND}"
    )

    # Load datasets
    datasets = load_datasets(SELECT_DATASETS, max_per_dataset=MAX_EXAMPLES_PER_DATASET)

    # Run experiments
    results = run_experiments(
        datasets,
        model_id=MODEL_ID,
        dry_run=DRY_RUN,
        device=DEVICE,
        layer_index=LAYER_TO_EXTRACT,
        probe_C=PROBE_C,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED,
        train_upper_bound=TRAIN_UPPER_BOUND,
        upper_bound_mode=UPPER_BOUND_MODE,
    )

    # Report results
    report_results(results)
