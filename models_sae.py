"""
models_sae.py

Handles the loading of LLMs (Transformers) and Sparse Autoencoders (SAE-Lens).
Also includes a local, PyTorch-native fallback SAE class just in case.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor

from config import CONFIG
from utils import print_info

# Conditional imports for HuggingFace libraries
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    AutoTokenizer = AutoModelForCausalLM = BitsAndBytesConfig = None

def load_model(model_id: str, ignore_mismatched: bool = True):
    """Downloads and loads the target LLM to the specified device."""
    if CONFIG.DRY_RUN:
        print_info(f"[LoadModel] DRY_RUN: skipping download for {model_id}")
        return None, None
        
    if AutoTokenizer is None: 
        raise RuntimeError("transformers package not available")
    
    print_info(f"[LoadModel] loading {model_id} to device {CONFIG.DEVICE}")
    
    # Use bfloat16 to save memory if on GPU, otherwise float32 for CPU
    dtype = torch.float32 if CONFIG.DEVICE == "cpu" else (getattr(torch, "bfloat16", torch.float16))
    
    # 4-Bit quantization drastically reduces VRAM requirements for massive models
    bnb = BitsAndBytesConfig(load_in_4bit=True) if CONFIG.USE_4BIT else None
    
    # Resolve HuggingFace access token for gated repos (like Google's Gemma)
    hf_kwargs = {"use_auth_token": CONFIG.HF_TOKEN} if CONFIG.HF_TOKEN else {}
    
    # Let Accelerate figure out the device map if using GPU
    device_map = "auto" if CONFIG.DEVICE.startswith("cuda") else None
    

    if model_id == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.chat_template = processor.chat_template
        tokenizer = processor.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, **hf_kwargs)
        if getattr(tokenizer, "pad_token", None) is None: 
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb, 
            device_map=device_map, 
            trust_remote_code=True, 
            torch_dtype=dtype, 
            ignore_mismatched_sizes=ignore_mismatched, 
            **hf_kwargs
        )
        
    # Only force to device if accelerate didn't already map it
    if device_map is None:
        try: 
            model.to(CONFIG.DEVICE)
        except Exception: 
            pass
            
    print_info(f"[LoadModel] loaded {model_id}")
    return tokenizer, model

def load_sae_model():
    """Loads a pre-trained Sparse Autoencoder from sae_lens."""
    if CONFIG.DRY_RUN or CONFIG.SAE_SOURCE == "none":
        print_info("[SAE Load] DRY_RUN or SAE disabled: skipping SAE load")
        return None
        
    try:
        from sae_lens import SAE
        print_info(f"[SAE Load] loading SAE release {CONFIG.SAE_RELEASE} id={CONFIG.SAE_ID}")
        
        # Updated SAE API: unpacks the config and sparsity cleanly
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=CONFIG.SAE_RELEASE, 
            sae_id=CONFIG.SAE_ID, 
            device=CONFIG.DEVICE
        )
        sae.eval()
        
        print_info(f"[SAE Load] successfully loaded Gemma SAE. Sparse Dim: {sae.cfg.d_sae}")
        return sae
    except Exception as e:
        print_info(f"[SAE Load] could not load gemma scope SAE: {e}")
        return None

# ==========================================
# Local Fallback SAE Architecture
# ==========================================

class LocalSparseAutoencoder(nn.Module):
    """
    A simple Pytorch SAE architecture used as a fallback if `sae_lens` isn't available
    or if you want to train a custom SAE locally on your own activation cache.
    """
    def __init__(self, input_dim, bottleneck_dim=256, hidden_mult=2):
        super().__init__()
        # Heuristic for middle layer sizing
        h = max(int(bottleneck_dim * hidden_mult), max(int(input_dim // 2), bottleneck_dim))
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h), 
            nn.ReLU(), 
            nn.Linear(h, bottleneck_dim), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, h), 
            nn.ReLU(), 
            nn.Linear(h, input_dim)
        )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z