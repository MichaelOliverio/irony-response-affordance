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
    "batch_size": 32,
    "max_retries": 3,
    "temperature": 1.0,
    "max_new_tokens": 512,
}
DATA_AUGMENTATION_MODELS = [
    'meta-llama/Llama-3.1-8B-Instruct',
    #'mistralai/Ministral-8B-Instruct-2410',
]
DATA_AUGMENTATION_OUTPUT_DIR = DATA_AUGMENTATION_DIR / "output"

PROMPT_MAPPING = {
    1: "First",
    2: "Second",
    3: "Third",
    4: "Fourth",
    5: "Fifth"
}

NUM_MAX_TURNS = 2
FROM_ONE_TO_MAX_TURNS = False # Se True, genera da 1 fino a NUM_MAX_TURNS, altrimenti solo NUM_MAX_TURNS