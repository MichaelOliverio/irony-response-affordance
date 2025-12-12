import json
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from src.config import RANDOM_SEED, DATASET_SPLIT_RATIOS

class DatasetLoader:
    @staticmethod
    def load_aggregated_dataset(file_path: str, verbose: bool = False) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_dataset = data["entries"]

        dataset = []
        for entry in raw_dataset:
            dataset_name = entry.get("dataset_name")
            annotations = entry.get("annotations", {})

            # Determina il label in base al dataset
            if dataset_name == "MultiPICo":
                label = annotations.get("aggregate_irony_label")
            else:  # dataset CSC
                label = annotations.get("aggregate_sarcasm_label")

            # Converti context_type in 1/0/null
            context_type_raw = entry.get("context_type")
            if context_type_raw is None:
                context_type = None
            else:
                context_type = 1 if context_type_raw == "non_neutral" else 0

            cleaned_entry = {
                "unique_id": entry.get("unique_id"),
                "source": entry.get("source"),
                "context_type": context_type,
                "language": entry.get("language"),
                "context": entry.get("context"),
                "post": entry.get("post"),
                "reply": entry.get("reply"),
                "dialogue": entry.get("dialogue"),
                "label": label,
            }

            dataset.append(cleaned_entry)
        
        train_set, val_set, test_set = DatasetLoader.balanced_split(dataset)
        if verbose:
            print(f"Loaded {len(dataset)} entries from {file_path}")
            print("Overall label distribution:")
            DatasetLoader.get_label_distribution(dataset)
            print(f"\nTrain: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")
            print("\nLabel distributions per split:")
            print("Train set:")
            DatasetLoader.get_label_distribution(train_set)
            print("\nValidation set:")
            DatasetLoader.get_label_distribution(val_set)
            print("\nTest set:")
            DatasetLoader.get_label_distribution(test_set)

        return train_set, val_set, test_set

    @staticmethod
    def balanced_split(dataset: List[Dict]):
        train_ratio = DATASET_SPLIT_RATIOS['train']
        val_ratio = DATASET_SPLIT_RATIOS['val']
        test_ratio = DATASET_SPLIT_RATIOS['test']

        # Estrai le label
        labels = [entry['label'] for entry in dataset]

        # Primo split train vs temp (val+test)
        train_set, temp_set, train_labels, temp_labels = train_test_split(
            dataset, labels, test_size=(1-train_ratio), stratify=labels, random_state=RANDOM_SEED
        )

        # Calcolo proporzioni relative per val/test sul temp
        val_size = val_ratio / (val_ratio + test_ratio)

        val_set, test_set = train_test_split(
            temp_set, test_size=(1-val_size), stratify=temp_labels, random_state=RANDOM_SEED
        )

        return train_set, val_set, test_set

    @staticmethod
    def get_label_distribution(dataset: List[Dict]) -> Dict[str, int]:
        labels = [entry["label"] for entry in dataset]
        label_counts = Counter(labels)
        total = len(labels)

        print("Label distribution:")
        for lbl, count in label_counts.items():
            percent = (count / total) * 100
            print(f"  {lbl}: {count} ({percent:.2f}%)")