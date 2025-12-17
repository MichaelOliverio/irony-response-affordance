from pathlib import Path
import torch

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiment2" / "output"
DATASET_DIR = PROJECT_ROOT / "experiment2" / "disaggregated_dataset"

RANDOM_SEED = 42

DATASET_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

DATASETS = {
    'MultiPICo': {
        'path': f"{DATASET_DIR}/MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered_with_semiaggregates.json",
    },
    'CSC': {
        'path': f"{DATASET_DIR}/CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered_with_semiaggregates.json",
    },
}

DECODER_SETTINGS = {
    'models': [
        #'meta-llama/Llama-3.2-1B-Instruct',
        'Qwen/Qwen2.5-7B-Instruct',
        'mistralai/Ministral-8B-Instruct-2410',
        'google/gemma-2-9b-it',
    ],
    'training_params': {
        # Parametri di training specifici per i modelli decoder
    },
    'inference_params': {
        'max_new_tokens': 256,
        'temperature': 0.7,
        'batch_size': 32,
        'dtype': torch.float16
    },
    'output_dir': OUTPUT_DIR / "decoders"
}