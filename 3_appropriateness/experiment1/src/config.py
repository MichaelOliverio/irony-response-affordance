from pathlib import Path
import torch

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiment1" / "output"
DATASET_DIR = PROJECT_ROOT / "datasets"

RANDOM_SEED = 42

DATASET_SPLIT_RATIOS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

DATASETS = {
    'MultiPICo': {
        'path': f"{DATASET_DIR}/MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered.json",
    },
    #'CSC': {
    #    'path': f"{DATASET_DIR}/CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered.json",
    #},
}

DECODER_SETTINGS = {
    'models': [
        'meta-llama/Llama-3.2-1B-Instruct',
    ],
    'training_params': {
        'lora' : {
            'r': 64, 
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'bias': "none",
            'task_type': "CAUSAL_LM",
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        'training_args': {
            'output_dir': OUTPUT_DIR / "decoders" / "fine_tuned_models",
            'num_train_epochs': 5,
            'per_device_train_batch_size': 32,
            'per_device_eval_batch_size': 32,
            'gradient_accumulation_steps': 1,
            'optim': 'adamw_torch',
            'eval_strategy': 'epoch',  # Valuta ad ogni epoca
            'save_strategy': 'epoch',  # Salva checkpoint ad ogni epoca
            'logging_steps': 100,
            'learning_rate': 2e-4,
            'weight_decay': 0.001,
            'fp16': False,
            'bf16': True,
            'max_grad_norm': 0.3,
            'max_steps': -1,
            'warmup_ratio': 0.03,
            'save_total_limit': None,  # Mantiene tutti i checkpoint (uno per epoca)
            'load_best_model_at_end': True,  # Carica il modello migliore alla fine
            'metric_for_best_model': 'eval_loss',  # Usa eval_loss per scegliere il migliore
            'greater_is_better': False,  # eval_loss: più basso è meglio
            'group_by_length': True,
            'lr_scheduler_type': 'cosine',
            'report_to': 'none',
        }
    },
    'inference_params': {
        'max_new_tokens': 256,
        'temperature': 0.7,
        'batch_size': 4,
        'torch_dtype': torch.float16
    },
    'output_dir': OUTPUT_DIR / "decoders"
}

ENCODER_SETTINGS = {
    'models': [
        #"EuroBERT/EuroBERT-210m",
        "FacebookAI/xlm-roberta-base"
    ],
    'training_params': {
        # Parametri di training specifici per i modelli encoder
    },
    'output_dir': OUTPUT_DIR / "encoders"
}