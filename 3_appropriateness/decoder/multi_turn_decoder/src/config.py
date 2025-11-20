"""Configurazione centrale per i path del progetto"""
from pathlib import Path

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_AUGMENTATION_DIR = PROJECT_ROOT / "2_data_augmentation"
AUGMENTEND_CORPUS_PATH = DATA_AUGMENTATION_DIR / "output"
OUTPUT_DIR = PROJECT_ROOT / "3_appropriateness" / "output"

DECODER_PARAMS = {
    'models': [
        'meta-llama/Llama-3.2-3B-Instruct',
        #'meta-llama/Llama-3.1-8B-Instruct',
        #'Qwen/Qwen2.5-7B-Instruct',
        #'mistralai/Ministral-8B-Instruct-2410',
    ],
    'max_new_tokens': 512,
    'temperature': 0.5,
    'batch_size': 8,
}