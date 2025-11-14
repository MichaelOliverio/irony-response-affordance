"""Configurazione centrale per i path del progetto"""
from pathlib import Path

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory principali
DATASETS_MERGE_DIR = PROJECT_ROOT / "1_datasets_merge"
DATA_AUGMENTATION_DIR = PROJECT_ROOT / "2_data_augmentation"
APPROPRIATENESS_DIR = PROJECT_ROOT / "3_appropriatness"

# Path specifici dei file
UNIFIED_CORPUS_PATH = DATASETS_MERGE_DIR / "output" / "unified_corpus.json"

# Directory di output
DATASETS_MERGE_OUTPUT_DIR = DATASETS_MERGE_DIR / "output"
DATASETS_MERGE_LOG_DIR = DATASETS_MERGE_DIR / "log"

# Data Augmentation Constants
DATA_AUGMENTATION_HYPERPARAMS = {
    "batch_size": 8,
    "max_retries": 3,
    "temperature": 0.5,
    "max_new_tokens": 1024,
}
DATA_AUGMENTATION_MODELS = [
    'meta-llama/Llama-3.2-3B-Instruct',
    #'meta-llama/Llama-3.1-8B-Instruct',
    #'mistralai/Mixtral-8x7B-Instruct-v0.1',
    #'Qwen/Qwen2.5-7B-Instruct',
]
DATA_AUGMENTATION_OUTPUT_DIR = DATA_AUGMENTATION_DIR / "output"

PROMPT_MAPPING = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth"
}

NUM_MAX_TURNS = 3