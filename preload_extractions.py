from probes import get_activations, load_eval_datasets, load_model, prepare_chat_for_followup
from probe_config import ExperimentConfig

import os
import torch

def main():
    config = ExperimentConfig()
    
    for model_name_hf in config.MODELS_TO_TEST:
        print(f"\n=== Processing Model: {model_name_hf} ===")
        # config.EVAL_PATH / model_name / ds_name / activations.pt
        eval_subdir = os.path.join(config.EVAL_PATH, model_name_hf.replace("/", "_"))
        os.makedirs(eval_subdir, exist_ok=True)

        tokenizer, model = load_model(model_name_hf)
        model_name = model.config._name_or_path

        try: target_layer = int(model.config.num_hidden_layers * config.LAYER_PERCENTILE)
        except: target_layer = int(model.config.text_config.num_hidden_layers * config.LAYER_PERCENTILE) 
        
        print(f"Extracting activations for model: {model_name}")
        # Load evaluation datasets
        eval_datasets = load_eval_datasets(config.EVAL_PATH)
        
        # reverse the order of the dataset, to have the largest ones last (for better progress tracking)
        eval_datasets = dict(reversed(list(eval_datasets.items())))
        for ds_name, ds_data in eval_datasets.items():
            # Accumulate activations first, then evaluate each probe
            print(" > Activations from unmodified chats...")
            X_test_full, y_test_full = get_activations(tokenizer, model, ds_data, target_layer, None)
            save_path_full = os.path.join(eval_subdir, ds_name+"_activations_full.pt")
            torch.save((X_test_full, y_test_full), save_path_full)

            # Follow-up probe needs special data preparation (Follow-up question appended)
            print(" > Activations from follow-up chats...")
            X_test_fup, y_test_fup = get_activations(tokenizer, model, ds_data, target_layer, prepare_chat_for_followup)
            save_path_fup = os.path.join(eval_subdir, ds_name+"_activations_fup.pt")
            torch.save((X_test_fup, y_test_fup), save_path_fup)
            
            print(f"Saved activations for dataset '{ds_name}' to '{eval_subdir}'")

if __name__ == "__main__":    
    main()    