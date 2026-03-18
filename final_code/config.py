"""
config.py

This module defines the global configuration for the Liars' Bench + SAE Probing pipeline.
By centralizing these parameters, we avoid circular imports and allow easy tweaking 
of paths, model sizes, and hyperparameters before submitting a SLURM job.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentConfig:
    # ==========================================
    # I/O Directories
    # ==========================================
    TRAIN_PATH: str = "../train_data_azaria_mitchell"       # Directory containing training CSVs
    EVAL_PATH: str = "../eval_data_liars_bench"             # Directory containing evaluation JSONL files
    OUTPUT_DIR: str = "./results_final/mistral"                    # Where plots and metric summaries are saved
    ACTIVATION_CACHE_DIR: str = "./activation_cache"       # Directory for HDF5/Memmap large tensor storage

    # ==========================================
    # Model Selection
    # ==========================================
    # List of HuggingFace model IDs to run in standard "Dense" mode (Path 1)
    LLM_MODELS_TO_TEST: List[str] = field(default_factory=lambda: ["mistralai/Mistral-Small-3.1-24B-Instruct-2503"])
    
    # List of HuggingFace model IDs to run through the Sparse Autoencoder pipeline (Path 2)
    SAE_MODELS_TO_TEST: List[str] = field(default_factory=lambda: [])

    # ==========================================
    # Data Sampling Options
    # ==========================================
    # Fraction of the train/eval datasets to use. 
    # Use 1.0 for the final run, or a small float (e.g., 0.005) for quick debugging.
    TRAIN_SAMPLE_PERCENTAGE: float = 1
    EVAL_SAMPLE_PERCENTAGE: float = 1

    # ==========================================
    # DolusChat Synthetic Data Integration
    # ==========================================
    INCLUDE_DOLUSCHAT_IN_TRAIN: bool = True     # Whether to append synthetic lies/truths to the training set
    DOLUSCHAT_SIZE: int = 1000                  # Number of samples to pull from DolusChat
    ONLY_ALLOWED_LIE_TYPES: bool = False         # If True, filters DolusChat samples to only include those with lie types in ALLOWED_LIE_TYPES
    ALLOWED_LIE_TYPES = {                       # All used lie types in DolusChat (for reference and filtering)
        "omission"
        # "exaggeration"
        # "fabrication"
    }
    
    # ==========================================
    # Sparse Autoencoder (SAE) Parameters
    # ==========================================
    SAE_SOURCE: str = "none"             # SAE backend: "none" | "gemma_scope"
    SAE_RELEASE: str = ""  # Specific SAE release version
    SAE_ID: str = ""          # Specific SAE model ID (defines target layer and sparsity)
    SAE_HIDDEN_DIM: int = 5120                    # Dummy dimension for local SAE fallback (if Gemma Scope isn't used)
    SAE_EPOCHS: int = 2                         # Epochs for training a local SAE (if applicable)
    SAE_BATCH: int = 2                          # Batch size for local SAE training
    SAE_L1_LAMBDA: float = 1e-3                 # L1 regularization penalty for local SAE sparsity
    SAE_TARGET_LAYER: int = 12                  # Explicit layer target if not derived automatically

    # ==========================================
    # Hardware & Inference Parameters
    # ==========================================
    DRY_RUN: bool = False                       # If True, generates mock data and skips LLM downloads
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT: bool = True                      # If True, loads LLM in 4-bit quantization (requires bitsandbytes)
    MAX_MODEL_LEN: int = 1024                   # Maximum context length for the tokenizer
    LLM_BATCH_SIZE: int = 8                     # Batch size for LLM forward passes (reduce if hitting OOM)

    # ==========================================
    # Probing Methodology
    # ==========================================
    LAYER_PERCENTILE: float = 0.66              # If layer isn't explicit, probes layer at this depth fraction (e.g., 66% deep)
    LEARNING_RATE: float = 1e-3                 # Default LR for PyTorch-based probes
    EPOCHS: int = 20                             # Default training epochs for probes
    BATCH_SIZE: int = 64                         # Default batch size for probes
    SEED: int = 42                              # Random seed for reproducible dataset splits and initialization

    TRAIN_ACTIVATION_METHOD: str = "train_token_wise"  # Extraction mode for training (e.g., last 5 tokens)
    EVAL_ACTIVATION_METHOD: str = "mean_pool"          # Extraction mode for evaluation (e.g., average over answer)

    SELECTED_PROBES: List[str] = field(default_factory=lambda: [
        "logistic", "tpc", "truth2d", "mass_mean", "followup", "upper_bound", "inlp"
    ])

    # ==========================================
    # Caching & Environment
    # ==========================================
    VERBOSE: bool = True                        # If True, prints timestamped logs
    HF_TOKEN = os.environ.get("HF_TOKEN", None) # HuggingFace token for gated models (like Gemma)

    # Set to True to force-delete old HDF5 files on startup. Highly recommended if you change dimensions/sampling.
    CLEAN_ACTIVATION_CACHE: bool = True      

    # ==========================================
    # Checkpointing
    # ==========================================
    CHECKPOINT_KEEP_LAST_K: Optional[int] = 5   # Keeps only the N most recent checkpoints per probe to save disk space
    ENABLE_CHECKPOINT_ROTATION: bool = True     # Enables the deletion of older checkpoints
    CHECKPOINT_DIR: str = "./results_final/checkpoints" # Directory to save probe weights (.pth)

# Instantiate a global config object to be imported by other modules
CONFIG = ExperimentConfig()
layer = int(CONFIG.LAYER_PERCENTILE * 100)
CONFIG.OUTPUT_DIR += f"/layer{layer}"
if not CONFIG.INCLUDE_DOLUSCHAT_IN_TRAIN:
    CONFIG.OUTPUT_DIR += "/no_DolusChat"
elif not CONFIG.ONLY_ALLOWED_LIE_TYPES:
    CONFIG.OUTPUT_DIR += "/all_DolusChat"
else:
    CONFIG.OUTPUT_DIR += f"/{CONFIG.ALLOWED_LIE_TYPES[0]}"

# Ensure critical output directories exist upon initialization
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG.ACTIVATION_CACHE_DIR, exist_ok=True)