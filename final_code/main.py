"""
main.py

Orchestrates the entire Liars' Bench probing pipeline.
Runs Path 1 (Dense LLM activations) and Path 2 (Sparse Autoencoder features)
sequentially, collecting metrics and saving plots along the way.
"""

import glob
import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CONFIG
from utils import print_info, show_device_info, calculate_extended_metrics, plot_roc_curves, plot_score_distribution
from data_utils import load_train_data, load_eval_datasets
from activations import extract_and_save_activations, load_saved_activations, activation_filename
from models_sae import load_model, load_sae_model

# Import all Probe configurations
from probes import (
    PyTorchLogisticProbe, MassMeanProbe, INLPProbe, 
    TruncatedPolynomialProbeWrapper, TruthUniversal2DProbeWrapper
)

def generate_report(all_results):
    """Converts the nested results dictionary into DataFrames, prints markdown, and saves CSVs."""
    print_info("=== LIARS' BENCH: MULTI-METRIC REPORT ===")
    
    metric_order = [
        "AUROC",
        "BALANCEDACCURACYAT1%FPR", "BALANCEDACCURACYAT0.1%FPR", "BALANCEDACCURACYAT0.01%FPR",
        "RECALL", "RECALLAT1%FPR", "RECALLAT0.1%FPR", "RECALLAT0.01%FPR",
        "FPRAT1%FPR"
    ]

    for probe_name in all_results:
        print(f"\n\n{'#'*30}\n   REPORT: {probe_name.upper()}\n{'#'*30}")
        for metric in metric_order:
            if metric not in all_results[probe_name]: continue

            df = pd.DataFrame.from_dict(all_results[probe_name][metric], orient='index')
            if df.empty: continue

            # Sort and calculate averages
            df = df.sort_index(axis=1)
            df["AVERAGE"] = df.mean(axis=1)
            col_means = df.mean(axis=0)
            grand_mean = df["AVERAGE"].mean()
            df.loc["AVERAGE"] = col_means
            df.loc["AVERAGE", "AVERAGE"] = grand_mean

            # Print to SLURM output log
            print(f"\n>>> METRIC: {metric} <<<")
            print(df.to_markdown(floatfmt=".4f"))

            # Save to CSV in results_final
            safe_metric = metric.replace("%", "pct")
            csv_path = os.path.join(CONFIG.OUTPUT_DIR, f"table_{probe_name}_{safe_metric}.csv")
            df.to_csv(csv_path)
            print_info(f"Saved summary table to {csv_path}")


def evaluate_model(model_name: str):
    """Executes the extraction and probing evaluation for a single model configuration."""
    tokenizer, model = load_model(model_name)
    sae_model = load_sae_model()
    
    # FIX: Use the exact SAE layer if running Sparse, otherwise calculate dynamically
    if sae_model is not None:
        target_layer = CONFIG.SAE_TARGET_LAYER
        print_info(f"[Evaluate] SAE detected. Using explicit target_layer={target_layer}")
    else:
        total_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 30))
        target_layer = int(total_layers * CONFIG.LAYER_PERCENTILE)
        print_info(f"[Evaluate] Dense model. Calculated target_layer={target_layer} (Percentile: {CONFIG.LAYER_PERCENTILE})")
    
    train_data = load_train_data(CONFIG.TRAIN_PATH)
    eval_datasets = load_eval_datasets(CONFIG.EVAL_PATH)
    
    if not train_data or not eval_datasets:
        print_info("[Evaluate] No train or eval data -> abort")
        return {}

    # =========================================================
    # Pre-Slice Train Data: Prevents massive initial extractions
    # =========================================================
    sample_frac = float(getattr(CONFIG, "TRAIN_SAMPLE_PERCENTAGE", 1.0))
    if sample_frac < 1.0 and len(train_data) > 0:
        rng = random.Random(CONFIG.SEED)
        keep_n = max(1, int(len(train_data) * sample_frac))
        train_data = rng.sample(train_data, keep_n)
        print_info(f"[Evaluate] Downsampled raw train data to {len(train_data)} samples BEFORE extraction.")

    results_per_probe = {}
    probes = {}

    if CONFIG.CLEAN_ACTIVATION_CACHE:
        print_info("[Cache] Cleaning activation cache as requested")
        for f in glob.glob(os.path.join(CONFIG.ACTIVATION_CACHE_DIR, "*")):
            try: os.remove(f)
            except Exception: pass

    # 1. Main Training Feature Extraction
    if any(p in CONFIG.SELECTED_PROBES for p in ["logistic","tpc","truth2d","mass_mean","inlp"]):
        act_base = activation_filename(model_name, "train", target_layer, CONFIG.TRAIN_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR)
        meta_path = act_base + ".meta.json"
        try:
            if os.path.exists(act_base) and os.path.exists(meta_path):
                print_info(f"[Evaluate] loading cached train activations: {act_base}")
                X_tr, y_tr, _, _, _ = load_saved_activations(model_name, "train", target_layer, CONFIG.TRAIN_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR)
            else:
                print_info(f"[Evaluate] extracting + caching train activations to: {act_base}")
                extract_and_save_activations(tokenizer, model, train_data, model_name, "train", target_layer, CONFIG.TRAIN_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR, batch_size=CONFIG.LLM_BATCH_SIZE, sae_model=sae_model)
                X_tr, y_tr, _, _, _ = load_saved_activations(model_name, "train", target_layer, CONFIG.TRAIN_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR)
        except Exception as e:
            print_info(f"[Evaluate] failed to load/extract train activations: {e}")
            X_tr, y_tr = np.array([]), np.array([])

        if len(X_tr) > 0:
            print_info(f"[Evaluate] train activations shape: {X_tr.shape} labels: {np.bincount(y_tr)}")
            if "logistic" in CONFIG.SELECTED_PROBES:
                pl = PyTorchLogisticProbe(X_tr.shape[-1])
                pl.fit(X_tr, y_tr, model_name=model_name)
                probes["logistic"] = pl
            if "tpc" in CONFIG.SELECTED_PROBES:
                pt = TruncatedPolynomialProbeWrapper(X_tr.shape[-1])
                pt.fit(X_tr, y_tr, model_name=model_name)
                probes["tpc"] = pt
            if "truth2d" in CONFIG.SELECTED_PROBES:
                pt2 = TruthUniversal2DProbeWrapper(X_tr.shape[-1])
                pt2.fit(X_tr, y_tr, model_name=model_name)
                probes["truth2d"] = pt2
            if "mass_mean" in CONFIG.SELECTED_PROBES:
                pm = MassMeanProbe()
                pm.fit(X_tr, y_tr)
                probes["mass_mean"] = pm
            if "inlp" in CONFIG.SELECTED_PROBES:
                pinlp = INLPProbe(max_iters=CONFIG.EPOCHS)
                pinlp.fit(X_tr, y_tr)
                probes["inlp"] = pinlp

    # 2. Follow-up Probe Extraction
    if "followup" in CONFIG.SELECTED_PROBES:
        base_fu = activation_filename(model_name, "train", target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
        try:
            if os.path.exists(base_fu) and os.path.exists(base_fu + ".meta.json"):
                X_fu, y_fu, _, _, _ = load_saved_activations(model_name, "train", target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
            else:
                extract_and_save_activations(tokenizer, model, train_data, model_name, "train", target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR, batch_size=CONFIG.LLM_BATCH_SIZE, sae_model=None)
                X_fu, y_fu, _, _, _ = load_saved_activations(model_name, "train", target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
        except Exception as e:
            X_fu, y_fu = np.array([]), np.array([])

        if len(X_fu) > 0:
            pf = PyTorchLogisticProbe(X_fu.shape[-1])
            pf.fit(X_fu, y_fu, model_name=model_name, checkpoint_prefix="followup")
            probes["followup"] = pf

    # 3. Evaluate Datasets
    for p_name in probes.keys():
        results_per_probe[p_name] = {}
    if "upper_bound" in CONFIG.SELECTED_PROBES:
        results_per_probe["upper_bound"] = {}

    for ds_name, ds_data in eval_datasets.items():
        if len(ds_data) < 2: continue

        # =========================================================
        # Pre-Slice Eval Data: Bound lower limit to prevent crashing
        # =========================================================
        eval_sample_frac = float(getattr(CONFIG, "EVAL_SAMPLE_PERCENTAGE", 1.0))
        if eval_sample_frac < 1.0:
            rng_ev = random.Random(CONFIG.SEED)
            num_ev = max(20, int(len(ds_data) * eval_sample_frac)) 
            num_ev = min(num_ev, len(ds_data))
            ds_data = rng_ev.sample(ds_data, num_ev)
            
        print_info(f"[Evaluate] dataset: {ds_name} processing {len(ds_data)} rows")

        dataset_roc_data = {}
        act_base = activation_filename(model_name, ds_name, target_layer, CONFIG.EVAL_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR)
        meta_path = act_base + ".meta.json"

        try:
            # Dimension verification (critical for switching between Dense and Sparse mode)
            expected_dim = sae_model.cfg.d_sae if sae_model else getattr(model.config, "hidden_size", getattr(model.config, "n_embd", getattr(model.config, "d_model", 2304)))
            cache_valid = False
            
            if os.path.exists(act_base) and os.path.exists(meta_path):
                with open(meta_path, "r") as f: meta = json.load(f)
                if meta["X_shape"][1] == expected_dim:
                    cache_valid = True
                else:
                    print_info(f"[Evaluate] Cache dim mismatch for {ds_name} ({meta['X_shape'][1]} vs {expected_dim}). Forcing rebuild.")
                    for f_del in [act_base, meta_path, act_base+".X.dat", act_base+".labels.dat", act_base+".sample_idx.dat", act_base+".token_pos.dat"]:
                        if os.path.exists(f_del):
                            try: os.remove(f_del)
                            except: pass

            if not cache_valid:
                extract_and_save_activations(tokenizer, model, ds_data, model_name, ds_name, target_layer, CONFIG.EVAL_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR, batch_size=CONFIG.LLM_BATCH_SIZE, sae_model=sae_model)
            
            X_test, y_test, _, _, _ = load_saved_activations(model_name, ds_name, target_layer, CONFIG.EVAL_ACTIVATION_METHOD, CONFIG.ACTIVATION_CACHE_DIR)

            # Evaluate Main Probes
            if len(X_test) > 0:
                for p_name in ["logistic","tpc","truth2d","mass_mean","inlp"]:
                    if p_name in probes:
                        scores = probes[p_name].predict_score(X_test)
                        metrics, plot_d = calculate_extended_metrics(y_test, scores)
                        results_per_probe.setdefault(p_name, {})[ds_name] = metrics
                        if plot_d: dataset_roc_data[p_name] = plot_d
                        try: plot_score_distribution(y_test, scores, p_name, ds_name, model_name)
                        except Exception: pass

            # Evaluate Followup Probe
            if "followup" in probes:
                fu_eval_base = activation_filename(model_name, ds_name, target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
                if not os.path.exists(fu_eval_base + ".meta.json"):
                    extract_and_save_activations(tokenizer, model, ds_data, model_name, ds_name, target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR, batch_size=CONFIG.LLM_BATCH_SIZE)
                X_fu_test, y_fu_test, _, _, _ = load_saved_activations(model_name, ds_name, target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
                
                if len(X_fu_test) > 0:
                    scores = probes["followup"].predict_score(X_fu_test)
                    metrics, plot_d = calculate_extended_metrics(y_fu_test, scores)
                    results_per_probe.setdefault("followup", {})[ds_name] = metrics
                    if plot_d: dataset_roc_data["Follow-up Probe"] = plot_d

            # Evaluate Upper Bound (Oracle)
            if "upper_bound" in CONFIG.SELECTED_PROBES:
                # We use the followup extractions as the basis for the upper bound
                X_ub, y_ub, _, _, _ = load_saved_activations(model_name, ds_name, target_layer, "followup_token", CONFIG.ACTIVATION_CACHE_DIR)
                
                # Robust Stratification Check
                if len(X_ub) >= 4 and len(np.unique(y_ub)) > 1:
                    unique, counts = np.unique(y_ub, return_counts=True)
                    can_stratify = np.min(counts) >= 2
                    
                    X_tr_ub, X_te_ub, y_tr_ub, y_te_ub = train_test_split(
                        X_ub, y_ub, test_size=0.5, random_state=42, 
                        stratify=y_ub if can_stratify else None
                    )
                    
                    ub_probe = PyTorchLogisticProbe(X_tr_ub.shape[-1])
                    ub_probe.fit(X_tr_ub, y_tr_ub, epochs=CONFIG.EPOCHS, model_name=model_name, checkpoint_prefix="upper_bound", dataset_name=ds_name)
                    
                    scores_ub = ub_probe.predict_score(X_te_ub)
                    metrics, plot_d = calculate_extended_metrics(y_te_ub, scores_ub)
                    results_per_probe.setdefault("upper_bound", {})[ds_name] = metrics
                    if plot_d: dataset_roc_data["Upper Bound"] = plot_d

            if dataset_roc_data:
                plot_roc_curves(dataset_roc_data, ds_name, model_name)

        except Exception as e:
            print_info(f"[Evaluate] Eval failed for {ds_name}: {e}")

    return results_per_probe

def main_pipeline():
    print_info("===================================================")
    print_info("=== START: Liars' Bench + SAE + Probes Pipeline ===")
    print_info("===================================================")
    show_device_info()
    all_results = {}
    
    # PATH 1: Dense LLM Probing
    if getattr(CONFIG, "LLM_MODELS_TO_TEST", []):
        print_info("\n" + "#" * 65)
        print_info(">>> STARTING PATH 1: DENSE LLM PROBING (ORIGINAL BASELINE) <<<")
        print_info("#" * 65 + "\n")
        
        original_sae_source = CONFIG.SAE_SOURCE
        CONFIG.SAE_SOURCE = "none" 
        for model_name in CONFIG.LLM_MODELS_TO_TEST:
            print_info(f"\n--- Running Dense model: {model_name} ---")
            res = evaluate_model(model_name)
            for probe, ds_dict in res.items():
                all_results.setdefault(probe, {})
                for ds_name, metrics in ds_dict.items():
                    for mkey, val in metrics.items():
                        all_results[probe].setdefault(mkey, {}).setdefault(f"{model_name}_Dense", {})[ds_name] = val
        CONFIG.SAE_SOURCE = original_sae_source

    # PATH 2: Sparse SAE Probing
    if getattr(CONFIG, "SAE_MODELS_TO_TEST", []):
        print_info("\n" + "=" * 65)
        print_info(">>> STARTING PATH 2: SPARSE SAE PROBING (APOLLO METHOD) <<<")
        print_info("=" * 65 + "\n")
        
        for model_name in CONFIG.SAE_MODELS_TO_TEST:
            print_info(f"\n--- Running SAE model pipeline: {model_name} + {CONFIG.SAE_ID} ---")
            res = evaluate_model(model_name)
            for probe, ds_dict in res.items():
                all_results.setdefault(probe, {})
                for ds_name, metrics in ds_dict.items():
                    for mkey, val in metrics.items():
                        all_results[probe].setdefault(mkey, {}).setdefault(f"{model_name}_SparseSAE", {})[ds_name] = val

    print_info("\n================================")
    print_info("=== FINISHED: Full Pipeline! ===")
    print_info("================================\n")
    return all_results

if __name__ == "__main__":
    final_results = main_pipeline()
    generate_report(final_results)