Here is the complete, master `README.md` file. It merges your old setup instructions, dependencies, repository links, and exact paper title with the new configuration guide, extended metrics explanation, and university acknowledgments.

```markdown
# Beyond Liars' Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### Work in Progress
Please note that this repository is currently a work in progress. It contains the source code and experimental pipeline supporting the paper **"Beyond Liars' Bench: The Impact of Lie Typology, Depth, and Sparsity on Deception Detection in LLMs"**. 

### Acknowledgments
We would like to extend our sincere thanks to the **University of Bonn** for the opportunity to conduct this research within the **EMA Lab**. We are also highly appreciative of the computational resources provided, as all experiments and model training for this project were executed on the university's **Bender cluster**.

---

## Quick Start / Local Setup 

Follow these steps to configure and set up the pipeline on your local environment.

### 1. Clone the Repository

```bash
git clone [https://github.com/amrgaber249/Beyond-Liars-Bench.git](https://github.com/amrgaber249/Beyond-Liars-Bench.git)
cd Beyond-Liars-Bench

```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

```

### 3. Activate the Environment

```bash
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

```

### 4. Install Dependencies

Ensure `pip` is updated to the latest version before installing dependencies:

```bash
python -m pip install --upgrade pip

```

#### Using a `requirements.txt` File

Execute:

```bash
pip install -r requirements.txt

```

---

## Extended Metrics

This repository includes additional evaluation metrics that were not featured in the original paper or the results directory due to time restrictions prior to publication. Specifically, the code evaluates the models at extremely strict False Positive Rates (FPR) to analyze performance under low-tolerance conditions.

These metrics are calculated dynamically during evaluation and printed directly to the console. They include:

* **AUROC** *(IN PAPER)*
* **Recall** (Standard) *(IN PAPER)*
* **Balanced Accuracy & Recall** at **1%**, **0.1%**, and **0.01% FPR**. *(NOT IN PAPER)*

---

## Configuration Guide (`config.py`)

All global parameters are centralized within `config.py`.

**File Location:** `config.py` is located in the root directory.

**Key Parameters to Adjust:**

* **Debugging and Testing:**
* `TRAIN_SAMPLE_PERCENTAGE` / `EVAL_SAMPLE_PERCENTAGE`: Reduce from `1` to a small float (e.g., `0.005`) to run the pipeline on a minimal data subset. This is highly recommended for quick compilation and debugging checks.
* `DRY_RUN`: Set to `True` to generate mock data and entirely bypass downloading large LLMs.


* **Model Selection:**
* `LLM_MODELS_TO_TEST`: Modify this list (e.g., `["google/gemma-2-27b"]`) for standard dense model runs.
* `SAE_MODELS_TO_TEST`: Update this list to route the pipeline through a Sparse Autoencoder.


* **Directory and Cache Management:**
* `OUTPUT_DIR`: Defines the save location for plots and metric summaries. The script automatically appends subdirectories based on layer percentile and DolusChat configurations.
* `CLEAN_ACTIVATION_CACHE`: Ensure this is set to `True` when altering dataset dimensions or sampling rates. This deletes obsolete HDF5 cache files on startup and prevents tensor shape mismatches.


* **Hardware Optimization (OOM Prevention):**
* `LLM_BATCH_SIZE`: Decrease this value (default is `8`) if the GPU encounters Out of Memory (OOM) errors during forward passes.
* `USE_4BIT`: Set to `True` to load models using BitsAndBytes quantization. Maintain `DEVICE = "cuda"` for execution on the cluster.


* **Probing and Synthetic Data Integration:**
* `INCLUDE_DOLUSCHAT_IN_TRAIN`: Toggle this boolean to control the integration of the additional doluschat data in the training set.
* `SELECTED_PROBES`: Adjust this list to test alternative probing methods (defaults include `logistic`, `tpc`, `inlp`, etc.).



---

## Example Configurations

Below are three configuration templates for `config.py`. Use these code blocks as a reference when switching environments or turning the Sparse Autoencoder pipeline on or off.

### 1. Mistral 24B (Dense Model Only)

Runs the Mistral model in standard dense mode. The SAE pipeline is disabled, and the synthetic DolusChat data is fully integrated without being filtered by specific lie types. This is running on layer 12 on Mistral which is at ~20% of the model architecture.

```python
    # Model Selection
    LLM_MODELS_TO_TEST: List[str] = field(default_factory=lambda: ["mistralai/Mistral-Small-3.1-24B-Instruct-2503"])
    SAE_MODELS_TO_TEST: List[str] = field(default_factory=lambda: []) # if it contains a model the model will also use the same config to run as a SAE model afterwards (not recommended)

    # DolusChat Synthetic Data Integration
    INCLUDE_DOLUSCHAT_IN_TRAIN: bool = True
    DOLUSCHAT_SIZE: int = 1000
    ONLY_ALLOWED_LIE_TYPES: bool = False

    # Sparse Autoencoder (SAE) Parameters (Disabled) (recommended to leave empty for SAE run)
    SAE_SOURCE: str = "none"
    SAE_RELEASE: str = ""
    SAE_ID: str = ""
    SAE_HIDDEN_DIM: int = 5120
    SAE_EPOCHS: int = 2
    SAE_BATCH: int = 2
    SAE_L1_LAMBDA: float = 1e-3
    SAE_TARGET_LAYER: int = 12

```

### 2. Gemma-2 27B (SAE Pipeline via Gemma Scope)

Routes the Gemma model activations through the SAE pipeline using Gemma Scope. The standard dense testing list is empty, and the training phase isolates specific allowed lie types (e.g., omission). This is running on layer 31 on Gemma 2 which is at ~66% of the model architecture.

```python
    # Model Selection
    LLM_MODELS_TO_TEST: List[str] = field(default_factory=lambda: [])
    SAE_MODELS_TO_TEST: List[str] = field(default_factory=lambda: ["google/gemma-2-27b"])

    # DolusChat Synthetic Data Integration
    INCLUDE_DOLUSCHAT_IN_TRAIN: bool = True
    DOLUSCHAT_SIZE: int = 1000
    ONLY_ALLOWED_LIE_TYPES: bool = True

    # Sparse Autoencoder (SAE) Parameters (Enabled)
    SAE_SOURCE: str = "gemma_scope"
    SAE_RELEASE: str = "gemma-scope-27b-pt-res-canonical"
    SAE_ID: str = "layer_31/width_131k/canonical"
    SAE_HIDDEN_DIM: int = 131072
    SAE_EPOCHS: int = 2
    SAE_BATCH: int = 2
    SAE_L1_LAMBDA: float = 1e-3
    SAE_TARGET_LAYER: int = 31

```

### 3. Gemma-2 27B (Dense Model Only)

Evaluates the same Gemma model variant but bypasses the SAE pipeline completely and analyze it as dense representations. This is also running on layer 31 on Gemma 2 which is at ~66% of the model architecture.

```python
    # Model Selection
    LLM_MODELS_TO_TEST: List[str] = field(default_factory=lambda: ["google/gemma-2-27b"])
    SAE_MODELS_TO_TEST: List[str] = field(default_factory=lambda: [])

    # Sparse Autoencoder (SAE) Parameters (Enabled)
    SAE_SOURCE: str = "gemma_scope"
    SAE_RELEASE: str = "gemma-scope-27b-pt-res-canonical"
    SAE_ID: str = "layer_31/width_131k/canonical"
    SAE_HIDDEN_DIM: int = 131072
    SAE_EPOCHS: int = 2
    SAE_BATCH: int = 2
    SAE_L1_LAMBDA: float = 1e-3
    SAE_TARGET_LAYER: int = 31

```

```

```