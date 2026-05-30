"""
activations.py

Handles the extraction of hidden states from LLMs and their projection into 
Sparse Autoencoder (SAE) feature spaces. Manages safe, chunked disk writing 
using HDF5 (or numpy memmap) to prevent Out-Of-Memory (OOM) errors.
"""

import os
import json
import errno
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict

from config import CONFIG
from utils import print_info, _ensure_dir
from data_utils import format_chat

# Prefer HDF5 for large chunked writes, fallback to memmap if unavailable
try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False

def _save_metadata(meta_path, meta_dict):
    """Saves a JSON metadata file alongside the tensor cache for easy dimension/shape tracking."""
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_dict, fh, indent=2)

def activation_filename(model_name: str, dataset_name: str, layer_idx: int, method: str, out_dir: str):
    """Generates a standardized filename for the activation cache."""
    safe_model = model_name.replace("/", "_")
    safe_ds = dataset_name.replace("/", "_")
    base = f"{safe_model}__{safe_ds}__layer{layer_idx}__{method}"
    return os.path.join(out_dir, base + ".h5") if HAS_H5 else os.path.join(out_dir, base)

def _estimate_rows_and_hidden_dim(tokenizer, data, layer_idx, method, model=None, batch_size=64, sae_model=None):
    """
    Performs a lightweight dry-run to count exactly how many rows of activations 
    we need to allocate on disk. This prevents dynamic resizing bottlenecks later.
    """
    total_rows = 0
    max_seq_len = 0
    n = len(data)
    
    for i in tqdm(range(0, n, batch_size), desc=f"Estimating extraction size", leave=False):
        batch = data[i:i+batch_size]
        texts = []
        for d in batch:
            if method == "followup_token":
                # Properly format the follow-up using the model's native chat template
                msgs = d["messages"].copy()
                msgs.append({"role": "user", "content": "Was your last response deceptive? (Yes/No)"})
                msgs.append({"role": "assistant", "content": "No"})
                texts.append(format_chat(tokenizer, msgs))
            else:
                texts.append(format_chat(tokenizer, d["messages"]))
            
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG.MAX_MODEL_LEN)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        
        if method == "train_token_wise":
            resp_lens = []
            for j, d in enumerate(batch):
                resp_txt = d["messages"][-1]["content"] if method != "followup_token" else "No"
                resp_len = len(tokenizer(resp_txt, add_special_tokens=False).input_ids)
                # Cap token-wise extraction to the last 5 tokens of the response
                resp_lens.append(min(5, resp_len))
            total_rows += sum(resp_lens)
        elif method in ("mean_pool", "last_token", "followup_token"):
            # These methods aggregate or pull exactly 1 vector per sample
            total_rows += len(batch)
            
        max_seq_len = max(max_seq_len, max(lengths))

    # Determine dimensions based on model config or SAE config
    hidden_dim = None
    if model is not None:
        hidden_dim = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", getattr(model.config, "d_model", None)))
    if sae_model is not None and hasattr(sae_model, "cfg"):
        hidden_dim = sae_model.cfg.d_sae

    return total_rows, max_seq_len, hidden_dim

def extract_and_save_activations(tokenizer, model, data: List[Dict], model_name: str, dataset_name: str, layer_idx: int, method: str, out_dir: str, batch_size: int = 16, dtype=np.float32, sae_model=None):
    """
    Core extraction engine. Runs data through the LLM, optionally projects through the SAE,
    and streams the resulting feature vectors safely into an HDF5 file on disk.
    """
    _ensure_dir(out_dir)
    fname = activation_filename(model_name, dataset_name, layer_idx, method, out_dir)
    meta_path = fname + ".meta.json"
    print_info(f"[ActSave] target base path: {fname}")

    # Safety check: if files already exist, skip processing to save time
    if os.path.exists(fname) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh: meta = json.load(fh)
        print_info(f"[ActSave] found existing activation file, skipping extraction: {fname}")
        return meta

    total_rows, max_seq_len, hidden_dim = _estimate_rows_and_hidden_dim(tokenizer, data, layer_idx, method, model=model, batch_size=batch_size, sae_model=sae_model)
    
    # Fallback to discover hidden_dim if config was obscured
    if hidden_dim is None:
        dummy_text = format_chat(tokenizer, data[0]["messages"]) if len(data) > 0 else "Hello"
        enc = tokenizer([dummy_text], return_tensors="pt", truncation=True, padding=True, max_length=CONFIG.MAX_MODEL_LEN).to(CONFIG.DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx]
            if sae_model is not None: hs = sae_model.encode(hs.to(sae_model.device))
            hidden_dim = int(hs.shape[-1])

    # Allocate slightly more space than estimated just in case of padding anomalies
    alloc_rows = max(total_rows, int(total_rows * 1.2) + 1000)
    print_info(f"[ActSave] estimated rows={total_rows}, allocating_rows={alloc_rows}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")

    # Initialize disk containers
    if HAS_H5:
        f = h5py.File(fname, "w")
        X_dset = f.create_dataset("X", shape=(alloc_rows, hidden_dim), maxshape=(None, hidden_dim), dtype='f4', chunks=True)
        labels_dset = f.create_dataset("labels", shape=(alloc_rows,), maxshape=(None,), dtype='i4', chunks=True)
        sample_idx_dset = f.create_dataset("sample_idx", shape=(alloc_rows,), maxshape=(None,), dtype='i4', chunks=True)
        token_pos_dset = f.create_dataset("token_pos", shape=(alloc_rows,), maxshape=(None,), dtype='i4', chunks=True)
    else:
        # Memmap fallback
        X_dset = np.memmap(fname + ".X.dat", dtype='float32', mode='w+', shape=(alloc_rows, hidden_dim))
        labels_dset = np.memmap(fname + ".labels.dat", dtype='int32', mode='w+', shape=(alloc_rows,))
        sample_idx_dset = np.memmap(fname + ".sample_idx.dat", dtype='int32', mode='w+', shape=(alloc_rows,))
        token_pos_dset = np.memmap(fname + ".token_pos.dat", dtype='int32', mode='w+', shape=(alloc_rows,))

    write_ptr = 0
    model.eval()
    
    try:
        for i in tqdm(range(0, len(data), batch_size), desc=f"Extracting {method}", leave=False):
            batch = data[i:i+batch_size]
            texts = []
            for d in batch:
                if method == "followup_token":
                    # Properly format the follow-up using the model's native chat template
                    msgs = d["messages"].copy()
                    msgs.append({"role": "user", "content": "Was your last response deceptive? (Yes/No)"})
                    msgs.append({"role": "assistant", "content": "No"})
                    texts.append(format_chat(tokenizer, msgs))
                else:
                    texts.append(format_chat(tokenizer, d["messages"]))
            
            # Send input to where the model resides (compatible with accelerate split models)
            target_device = model.device if hasattr(model, "device") else CONFIG.DEVICE
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=CONFIG.MAX_MODEL_LEN)
            enc = {k: v.to(target_device) for k, v in enc.items()}

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer_idx]
                
                # Apply SAE if running in Sparse Mode
                if sae_model is not None and hasattr(sae_model, "encode"):
                    try:
                        # Match SAE's specific device and dtype requirements explicitly
                        sae_device = getattr(sae_model, "device", target_device)
                        sae_dtype = getattr(sae_model, "dtype", torch.float32)
                        hs = sae_model.encode(hs.to(device=sae_device, dtype=sae_dtype))
                    except Exception as e:
                        raise RuntimeError(f"CRITICAL SAE ENCODE FAILURE: {e}")
                        
                # Cast to standard float32 for stable disk writing
                hs = hs.to(torch.float32) if hs.dtype != torch.float32 else hs.float()

            att_mask = enc["attention_mask"].cpu().numpy()
            
            # Parse responses to identify exactly which tokens to extract
            for j, item in enumerate(batch):
                valid_len = int(att_mask[j].sum())
                resp_txt = item["messages"][-1]["content"] if method != "followup_token" else "No"
                resp_len = len(tokenizer(resp_txt, add_special_tokens=False).input_ids)
                
                if method in ("last_token", "followup_token"):
                    idx = max(0, valid_len - 1)
                    vec = hs[j, idx:idx+1, :].cpu().numpy()
                    rows_needed = 1
                elif method == "mean_pool":
                    start = max(0, valid_len - resp_len)
                    vec = hs[j, start:valid_len, :].mean(dim=0, keepdim=True).cpu().numpy() if valid_len > start else hs[j, max(0, valid_len-1):valid_len, :].mean(dim=0, keepdim=True).cpu().numpy()
                    rows_needed = 1
                elif method == "train_token_wise":
                    start, end = max(0, valid_len - resp_len), max(max(0, valid_len - resp_len), valid_len - 5)
                    if end > start:
                        vec = hs[j, start:end, :].cpu().numpy()
                        rows_needed = vec.shape[0]
                    else: continue

                # Dynamic growth if we underestimated rows
                if write_ptr + rows_needed > (X_dset.shape[0]):
                    new_target = max(X_dset.shape[0] * 2, write_ptr + rows_needed + 1000)
                    if HAS_H5:
                        X_dset.resize((new_target, hidden_dim))
                        labels_dset.resize((new_target,))
                        sample_idx_dset.resize((new_target,))
                        token_pos_dset.resize((new_target,))
                    else:
                        raise NotImplementedError("Dynamic resizing of memmap not supported in modular mode. Use HDF5.")

                # Perform Disk Write
                if rows_needed == 1:
                    X_dset[write_ptr:write_ptr+1, :] = vec
                    labels_dset[write_ptr] = int(item["label"])
                    sample_idx_dset[write_ptr] = i + j
                    token_pos_dset[write_ptr] = -1 if method == "mean_pool" else (idx if method in ("last_token","followup_token") else -1)
                    write_ptr += 1
                else:
                    T = rows_needed
                    X_dset[write_ptr:write_ptr+T, :] = vec
                    labels_dset[write_ptr:write_ptr+T] = int(item["label"])
                    sample_idx_dset[write_ptr:write_ptr+T] = i + j
                    token_pos_dset[write_ptr:write_ptr+T] = np.arange(start, end, dtype=np.int32)
                    write_ptr += T
                    
    finally:
        # GUARANTEE release of file locks even if extraction crashes mid-way
        if not HAS_H5: 
            X_dset.flush(); labels_dset.flush(); sample_idx_dset.flush(); token_pos_dset.flush()
        else:
            if 'f' in locals():
                try: f.close()
                except Exception: pass

    final_shape = (write_ptr, hidden_dim)
    
    # Shrink HDF5 file exactly to the size of data written
    if HAS_H5:
        f = h5py.File(fname, "r+")
        if write_ptr < f["X"].shape[0]:
            f["X"].resize((write_ptr, hidden_dim))
            f["labels"].resize((write_ptr,))
            f["sample_idx"].resize((write_ptr,))
            f["token_pos"].resize((write_ptr,))
        f.close()

    # Save meta map
    meta = {"file": fname, "X_shape": list(final_shape), "dtype": str(dtype), "written_rows": int(final_shape[0]), "model_name": model_name, "dataset_name": dataset_name, "layer_idx": int(layer_idx), "method": method}
    _save_metadata(meta_path, meta)
    print_info(f"[ActSave] extraction complete, rows={meta['written_rows']}, saved to {fname}")
    return meta

def load_saved_activations(model_name: str, dataset_name: str, layer_idx: int, method: str, out_dir: str):
    """Loads cached features from disk based on the dataset and layer configuration."""
    fname = activation_filename(model_name, dataset_name, layer_idx, method, out_dir)
    meta_path = fname + ".meta.json"
    if not (os.path.exists(fname) or os.path.exists(fname + ".X.dat")) or not os.path.exists(meta_path):
        raise FileNotFoundError(errno.ENOENT, "Activation file not found", fname)
        
    with open(meta_path, "r", encoding="utf-8") as fh: 
        meta = json.load(fh)
        
    rows, dim = meta["X_shape"]
    if HAS_H5:
        f = h5py.File(fname, "r")
        X = f["X"][:]
        labels = f["labels"][:]
        sample_idx = f["sample_idx"][:]
        token_pos = f["token_pos"][:]
        f.close()
    else:
        X = np.memmap(fname + ".X.dat", dtype='float32', mode='r', shape=(rows, dim))[:]
        labels = np.memmap(fname + ".labels.dat", dtype='int32', mode='r', shape=(rows,))[:]
        sample_idx = np.memmap(fname + ".sample_idx.dat", dtype='int32', mode='r', shape=(rows,))[:]
        token_pos = np.memmap(fname + ".token_pos.dat", dtype='int32', mode='r', shape=(rows,))[:]
        
    return X, labels, sample_idx, token_pos, meta