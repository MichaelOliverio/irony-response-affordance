import json
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import (
    HUMAN_ANNOTATION_SETTINGS,
    RANDOM_SEED,
    DATASET_SPLIT_RATIOS,
)

class DatasetLoader:
    @staticmethod
    def load_cleaned_dataset(file_path: str) -> List[Dict]:
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
        
        return dataset

    @staticmethod
    def load_aggregated_dataset(file_path: str, verbose: bool = False) -> List[Dict]:
        dataset = DatasetLoader.load_cleaned_dataset(file_path)
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
    def load_human_majority_labels(dataset_name: str) -> Dict[str, int]:
        """Reproduce the majority-vote labels built in human/ann_analysis.ipynb."""
        settings = HUMAN_ANNOTATION_SETTINGS
        accepted_labels = set(settings['label_mapping'])
        annotator_frames = []
        invalid_ids = set()

        for annotator in settings['annotators']:
            annotation_path = settings['directory'] / f"{annotator}.xlsx"
            frame = pd.read_excel(annotation_path, sheet_name=dataset_name)
            frame = frame.rename(columns={settings['label_column']: 'human_label'})

            required_columns = {'unique_id', 'human_label'}
            missing_columns = required_columns.difference(frame.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing columns {sorted(missing_columns)} in {annotation_path} "
                    f"(sheet {dataset_name})."
                )
            if frame['unique_id'].duplicated().any():
                raise ValueError(
                    f"Duplicate unique_id values in {annotation_path} "
                    f"(sheet {dataset_name})."
                )

            frame['unique_id'] = frame['unique_id'].astype(str)
            invalid_ids.update(
                frame.loc[
                    ~frame['human_label'].isin(accepted_labels), 'unique_id'
                ].tolist()
            )
            annotator_frames.append(frame[['unique_id', 'human_label']])

        common_ids = set(annotator_frames[0]['unique_id'])
        for frame in annotator_frames[1:]:
            common_ids.intersection_update(frame['unique_id'])
        valid_ids = sorted(common_ids.difference(invalid_ids))

        labels = {}
        for unique_id in valid_ids:
            votes = [
                frame.loc[frame['unique_id'] == unique_id, 'human_label'].iloc[0]
                for frame in annotator_frames
            ]
            positive_votes = sum(label == 'triggering' for label in votes)
            labels[unique_id] = int(positive_votes >= (len(votes) // 2 + 1))

        return labels

    @staticmethod
    def load_human_annotated_ids(dataset_name: str) -> set[str]:
        """Return every annotated ID, including rows with invalid/missing labels."""
        settings = HUMAN_ANNOTATION_SETTINGS
        annotated_ids = set()

        for annotator in settings['annotators']:
            annotation_path = settings['directory'] / f"{annotator}.xlsx"
            frame = pd.read_excel(
                annotation_path,
                sheet_name=dataset_name,
                usecols=['unique_id'],
            )
            annotated_ids.update(frame['unique_id'].dropna().astype(str))

        return annotated_ids

    @staticmethod
    def build_human_evaluation_set(
        dataset: List[Dict], dataset_name: str
    ) -> List[Dict]:
        majority_labels = DatasetLoader.load_human_majority_labels(dataset_name)
        entries_by_id = {str(entry['unique_id']): entry for entry in dataset}
        missing_ids = sorted(set(majority_labels).difference(entries_by_id))
        if missing_ids:
            print(
                f"Warning: excluding {len(missing_ids)} human-annotated IDs "
                f"that are absent from the redundancy-filtered {dataset_name} "
                f"dataset; examples: {missing_ids[:5]}"
            )

        evaluation_set = []
        for unique_id, human_label in majority_labels.items():
            if unique_id not in entries_by_id:
                continue
            entry = dict(entries_by_id[unique_id])
            entry['human_majority_label'] = human_label
            evaluation_set.append(entry)
        return evaluation_set

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
