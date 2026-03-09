"""
data_utils.py

Handles loading, parsing, and cleaning of training and evaluation datasets.
Integrates HuggingFace Datasets for the DolusChat synthetic dataset.
"""

import os
import glob
import json
import random
import re
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

from config import CONFIG
from utils import print_info

# Optional HF Dataset import
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def format_chat(tokenizer, messages: List[Dict]) -> str:
    """Formats chat dictionaries into a string according to the model's chat template."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    # Fallback formatting if tokenizer lacks a template
    text = ""
    for m in messages:
        text += f"{m['role'].upper()}: {m['content']}\n"
    return text

def clean_doluschat_response(text: str) -> str:
    """
    Strips predictable prefixes and brackets from synthetic responses.
    Prevents the probe from just learning to detect the phrase "to be honest" instead of the concept of deception.
    """
    if not isinstance(text, str): return ""
    text = re.sub(r'\[.*?\]', '', text)
    text = text.strip(' \n\r"\'')
    
    # Remove common AI "tells"
    openers = [r"^i regret to inform you that ", r"^without a doubt,? i would say ", r"^to be honest,? ", r"^actually,? ", r"^well,? "]
    for op in openers:
        text = re.sub(op, '', text, flags=re.IGNORECASE)
        
    text = text.strip()
    if text: 
        text = text[0].upper() + text[1:]
    return text

def load_doluschat_data(n: int = 500, randomize_keep: bool = True, keep_prob: float = 0.5, enforce_balance: bool = True) -> List[Dict]:
    """Loads and balances synthetic truthful and deceptive data from the DolusChat HuggingFace repo."""
    if not CONFIG.INCLUDE_DOLUSCHAT_IN_TRAIN:
        print_info("[DolusChat] disabled by config")
        return []
        
    print_info("Loading AlignmentResearch/DolusChat (may take time if not in DRY_RUN)...")
    
    if CONFIG.DRY_RUN or load_dataset is None:
        out = []
        for i in range(min(n, 20)):
            out.append({"messages":[{"role":"user","content":"Tell me a fact."},{"role":"assistant","content":f"Mock factual true {i}"}],"label":0})
            out.append({"messages":[{"role":"user","content":"Tell me a fact."},{"role":"assistant","content":f"Mock deceptive lie {i}"}],"label":1})
        return out[:n]
        
    try:
        ds = load_dataset("AlignmentResearch/DolusChat", split="train")
    except Exception as e:
        print_info(f"Failed to load DolusChat: {e}")
        return []
        
    # We only want factual lies (not "I am a human" identity lies)
    allowed_lie_types = {"fabrication", "denial", "contradiction", "direct", "factual"}
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    target_true = n // 2 if enforce_balance else n
    target_false = n - target_true if enforce_balance else n
    count_true = 0; count_false = 0
    out = []
    
    for idx in tqdm(indices, desc="Filtering DolusChat Data", leave=False):
        if len(out) >= n: break
        
        row = ds[idx]
        if isinstance(row, dict): 
            lie_type = str(row.get("lie_type", "")).lower()
        else:
            row = dict(row)
            lie_type = str(row.get("lie_type", "")).lower()
            
        if CONFIG.ONLY_FACTUAL_STATEMENTS:
            if not any(a in lie_type for a in allowed_lie_types): continue
            
        truthful_raw = row["responses"].get("truthful", "") if "responses" in row and isinstance(row["responses"], dict) else row.get("truthful", "")
        deceptive_raw = row["responses"].get("deceptive", "") if "responses" in row and isinstance(row["responses"], dict) else row.get("deceptive", "")
        
        truthful_clean = clean_doluschat_response(truthful_raw)
        deceptive_clean = clean_doluschat_response(deceptive_raw)
        
        if truthful_clean and (count_true < target_true):
            if (not randomize_keep) or (random.random() < keep_prob):
                out.append({"messages":[{"role":"user","content":"Tell me a fact."},{"role":"assistant","content":truthful_clean}], "label":0})
                count_true += 1
                
        if deceptive_clean and (count_false < target_false):
            if (not randomize_keep) or (random.random() < keep_prob):
                out.append({"messages":[{"role":"user","content":"Tell me a fact."},{"role":"assistant","content":deceptive_clean}], "label":1})
                count_false += 1
                
    print_info(f"Loaded {len(out)} DolusChat samples (true={count_true}, false={count_false})")
    return out

def load_train_data(path_str: str) -> List[Dict]:
    """Loads all CSVs from the train directory and appends DolusChat data."""
    print_info("Loading training data (local csvs + optional DolusChat)...")
    if CONFIG.DRY_RUN:
        out = [{"messages":[{"role":"user","content":"Tell me a fact."},{"role":"assistant","content":f"Mock statement {i}"}],"label":i%2} for i in range(100)]
        return out
        
    all_data = []
    if os.path.exists(path_str):
        files = sorted(glob.glob(os.path.join(path_str, "*.csv")))
        for f in tqdm(files, desc="Loading Train CSVs", leave=False):
            try:
                df = pd.read_csv(f)
                df.columns = [c.lower().strip() for c in df.columns]
                if 'statement' not in df.columns or 'label' not in df.columns: continue
                for _, row in df.iterrows():
                    all_data.append({
                        "messages": [{"role":"user","content":"Tell me a fact."},
                                     {"role":"assistant","content": str(row.get('statement',''))}], 
                        "label": int(row['label'])
                    })
            except Exception: 
                continue
                
    # Append the synthetic data
    dc = load_doluschat_data(n=CONFIG.DOLUSCHAT_SIZE)
    if dc:
        print_info(f"Appending DolusChat: {len(dc)} samples to train set: {len(all_data)}")
        all_data.extend(dc)
        
    print_info(f"Full train data size: {len(all_data)}")
    return all_data

def load_eval_datasets(path_str: str) -> Dict[str, List[Dict]]:
    """Loads JSONL formatted eval datasets and maps true/false strings to binary labels."""
    print_info("Loading evaluation datasets...")
    if CONFIG.DRY_RUN: 
        return {"synthetic_eval": [{"messages":[{"role":"assistant","content":"Mock eval response"}], "label": i % 2} for i in range(200)]}
        
    datasets = {}
    if not os.path.exists(path_str): return datasets
    
    files = sorted(glob.glob(os.path.join(path_str, "*.jsonl")))
    for f in tqdm(files, desc="Loading Eval Datasets", leave=False):
        name = os.path.basename(f).replace(".jsonl","")
        data = []
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    
                    # Look for either 'deceptive' or 'label' keys
                    is_lie = obj.get("deceptive", obj.get("label"))
                    if is_lie is None: continue
                    
                    label = 1 if str(is_lie).lower() in ["true","1"] else 0
                    data.append({"messages": obj.get("messages",[]), "label": label})
                    
            if data: 
                # Note: The EVAL_SAMPLE_PERCENTAGE downsampling is now robustly handled 
                # inside main.py to prevent edge cases with tiny datasets. 
                # We return the full parsed list here.
                datasets[name] = data
        except Exception: 
            continue
            
    return datasets