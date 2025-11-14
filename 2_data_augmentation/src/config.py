"""Configurazione centrale per i path del progetto"""
from pathlib import Path

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory principali
DATASETS_MERGE_DIR = PROJECT_ROOT / "1_datasets_merge"
DATA_AUGMENTATION_DIR = PROJECT_ROOT / "2_data_augmentation"
APPROPRIATENESS_DIR = PROJECT_ROOT / "3_appropriatness"
MY_NOTES_DIR = PROJECT_ROOT / "my_notes"
NOTES_DIR = PROJECT_ROOT / "notes"

# Path specifici dei file
UNIFIED_CORPUS_PATH = DATASETS_MERGE_DIR / "output" / "unified_corpus.json"
UNIFIED_CORPUS_SAMPLE_PATH = DATASETS_MERGE_DIR / "output" / "unified_corpus_sample_10.json"

# Directory di output
DATASETS_MERGE_OUTPUT_DIR = DATASETS_MERGE_DIR / "output"
DATASETS_MERGE_LOG_DIR = DATASETS_MERGE_DIR / "log"

# Data Augmentation Constants
ACCEPTED_INPUT_TYPES = ["single-turn", "contextual"]
ACCEPTED_OUTPUT_TYPES = ["multi-turn"]
DATA_AUGMENTATION_HYPERPARAMS = {
    "batch_size": 1, #8
    "max_retries": 3,
    "temperature": 0.1,
    "max_new_tokens": 1024,
}
DATA_AUGMENTATION_MODELS = [
    'meta-llama/Llama-3.2-3B-Instruct',
    #'Qwen/Qwen2.5-3B-Instruct',
]
DATA_AUGMENTATION_OUTPUT_DIR = DATA_AUGMENTATION_DIR / "output"

def print_config():
    """Stampa la configurazione dei path"""
    print("=== Configurazione Path del Progetto ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"\nDirectory principali:")
    print(f"  DATASETS_MERGE_DIR: {DATASETS_MERGE_DIR}")
    print(f"  DATA_AUGMENTATION_DIR: {DATA_AUGMENTATION_DIR}")
    print(f"  APPROPRIATENESS_DIR: {APPROPRIATENESS_DIR}")
    print(f"\nFile principali:")
    print(f"  UNIFIED_CORPUS_PATH: {UNIFIED_CORPUS_PATH}")
    print(f"    Esiste: {UNIFIED_CORPUS_PATH.exists()}")
    print(f"  UNIFIED_CORPUS_SAMPLE_PATH: {UNIFIED_CORPUS_SAMPLE_PATH}")
    print(f"    Esiste: {UNIFIED_CORPUS_SAMPLE_PATH.exists()}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
