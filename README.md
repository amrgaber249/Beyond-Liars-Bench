# Liars' Bench — Deception Detection Pipeline

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]

A research-oriented pipeline to extract hidden activations from LLMs, cache them, train multiple probes (linear, low-rank polynomial, Truth2D, INLP, SAE-based features, Apollo follow-up trick), and evaluate on Liars' Bench-style datasets. Designed to be reproducible, checkpoint-friendly, and usable at small scale for quick experiments.

---

## Quick Start (Notebooks)

Run this in a Jupyter / Colab notebook cell. This will install the runtime dependencies, including `bitsandbytes` and `sae-lens` (from GitHub). 

> **Important:** `bitsandbytes` requires a supported CUDA toolkit / GPU runtime. If you run on CPU-only, remove it from the command.

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn h5py joblib accelerate safetensors bitsandbytes sae-lens
```

---

## Local Setup

Follow these steps to set up the pipeline on your local machine.

**1. Clone the repository**

```bash
git clone https://github.com/amrgaber249/Lab-AI-Alignment.git
cd Lab-AI-Alignment
```

**2. Create a virtual environment**

```bash
python -m venv .venv

```

**3. Activate the environment**

```bash
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

```

**4. Install Dependencies**
You can install the dependencies either via a quick one-line command OR by using a `requirements.txt` file. First, always make sure pip is up to date:

```bash
python -m pip install --upgrade pip

```

**Option A: Direct One-Line Install (Quickest)**

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn h5py joblib accelerate safetensors bitsandbytes sae-lens

```

**Option B: Using `requirements.txt**`
Create a `requirements.txt` file in your root directory containing:

```text
torch>=1.13.0
transformers>=4.30.0
datasets>=2.12.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.6.0
seaborn>=0.12.0
h5py>=3.8.0
joblib>=1.2.0
accelerate>=0.20.0
safetensors>=0.3.0
bitsandbytes>=0.39.0
sae-lens>=6.37.6

```

Then run:

```bash
pip install -r requirements.txt

```

**5. Run the pipeline**
