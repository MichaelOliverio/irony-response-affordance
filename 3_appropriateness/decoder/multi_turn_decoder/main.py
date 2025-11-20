import re
import pandas as pd
import json

from src.config import AUGMENTEND_CORPUS_PATH
from src.decoder_cot_classifier import DecoderCoTClassifier

def parse_filename(filename):
    pattern = re.compile(
        r'^(?P<dataset>[^_]+)_(?P<model>.+?)_num_gen_turn_(?P<turn>\d+)\.json$'
    )
    
    match = pattern.match(filename)
    if match:
        return match.group("dataset"), match.group("model"), int(match.group("turn"))
    else:
        return None

def main():
    folder_path = AUGMENTEND_CORPUS_PATH

    print(f"Processing files in: {folder_path}")
    for file_path in folder_path.glob("*.json"):
        print(f"\nProcessing file: {file_path.name}")
        parsed = parse_filename(file_path.name)
        if not parsed:
            print(f"Filename {file_path.name} does not match the expected pattern.")
            continue

        dataset, model_name, added_turn = parsed
        print(f"Dataset: {dataset}, Model: {model_name}, Added Turn: {added_turn}")

        # check if exists and open unified corpus in 1_datasets_merge\output\unified_corpus.jsonù
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                unified_corpus = json.load(f)
            print(f"Loaded unified corpus with {len(unified_corpus)} entries")
            df = pd.DataFrame(unified_corpus)
        except Exception as e:
            print(f"Error loading unified corpus: {e}")
            raise e

        DecoderCoTClassifier.compute_predictions(df, dataset, model_name, added_turn)

if __name__ == "__main__":
    main()
