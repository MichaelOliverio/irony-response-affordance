from pathlib import Path

# Root del progetto (questo file è nella root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "3_appropriateness" / "experiment1" / "output"

DATASETS = {
    'MultiPICo': {
        'baseline': {
            'context_field': 'post',
            'response_field': 'reply'
        },
        'ground_truth_field': 'aggregate_irony_label',
    },
    'CSC': {
        'baseline': {
            'context_field': 'context',
            'response_field': 'reply'
        },
        'ground_truth_field': 'aggregate_sarcasm_label',
    },
}

DECODER_MODELS = [
    # valutare quali usare
    'meta-llama/Llama-3.2-1B-Instruct',
]
ENCODER_MODELS = [
    # valutare quali usare
    "EuroBERT/EuroBERT-210m",
    "FacebookAI/xlm-roberta-base"
]