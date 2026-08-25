"""Evaluate Experiment 1 predictions against dataset and human references."""

import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
ENCODER_SPLIT_PROTOCOL = "human_heldout_f1"

DATASET_LABELS = {
    "CSC": "CANTRIP CSC",
    "MultiPICo": "CANTRIP MP",
}

MODEL_LABELS = {
    "Qwen_Qwen3.5-9B": "Qwen-3.5-9B",
    "mistralai_Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
    "google_gemma-4-E4B-it": "Gemma-4-E4B",
    "FacebookAI_xlm-roberta-large": "XLM-RoBERTa-large",
    "EuroBERT_EuroBERT-210m": "EuroBERT-210M",
    #"EuroBERT_EuroBERT-610m": "EuroBERT-610M",
}

INPUT_TYPES = (
    "context_reply",
    "last_u1_message",
    "conversation",
)


def readable_path(file_path: Path) -> str:
    """Return a path that also supports paths longer than 260 chars on Windows."""
    path = str(file_path.resolve())
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path


def parse_prediction_info(file_path: Path) -> dict:
    """Extract model, dataset, input, method, and reference from a path."""
    stem = file_path.stem
    dataset_name = next(
        (
            dataset
            for dataset in DATASET_LABELS
            if f"_{dataset}_" in stem
        ),
        None,
    )
    if dataset_name is None:
        raise ValueError("dataset name not found in filename")

    input_type = next(
        (input_name for input_name in INPUT_TYPES if input_name in stem),
        None,
    )
    if input_type is None:
        raise ValueError("input type not found in filename")

    if stem.startswith("zero_shot_"):
        technique = "Zero-Shot"
        technique_short = "ZS"
    elif stem.startswith("chain_of_thought_"):
        technique = "Chain-of-Thought"
        technique_short = "CT"
    elif stem.startswith("fine_tuning_"):
        technique = "Fine-Tuning"
        technique_short = "FT"
    else:
        raise ValueError("prompting/training method not found in filename")

    relative_parts = file_path.relative_to(OUTPUT_DIR).parts
    if len(relative_parts) < 3:
        raise ValueError("unexpected output path")
    model_name = relative_parts[1]

    is_human_cohort = stem.endswith("_human_majority")
    return {
        "dataset": dataset_name,
        "dataset_label": DATASET_LABELS[dataset_name],
        "model": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "input_type": input_type,
        "technique": technique,
        "technique_short": technique_short,
        "cohort": (
            "Human held-out"
            if is_human_cohort
            else "Standard test"
        ),
    }


def valid_evaluation_ids(frame: pd.DataFrame, reference_column: str) -> set[str]:
    """Return IDs with both a binary prediction and a binary reference."""
    required_columns = {"unique_id", "model_answer", reference_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing columns: {sorted(missing_columns)}")

    unique_ids = frame["unique_id"].astype("string")
    prediction_values = pd.to_numeric(frame["model_answer"], errors="coerce")
    reference_values = pd.to_numeric(frame[reference_column], errors="coerce")
    valid_mask = (
        unique_ids.notna()
        & prediction_values.isin([0, 1])
        & reference_values.isin([0, 1])
    )
    return set(unique_ids.loc[valid_mask].tolist())


def compute_metrics(frame: pd.DataFrame, reference_column: str) -> dict:
    required_columns = {"model_answer", reference_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing columns: {sorted(missing_columns)}")

    prediction_values = pd.to_numeric(
        frame["model_answer"], errors="coerce"
    )
    reference_values = pd.to_numeric(
        frame[reference_column], errors="coerce"
    )
    reference_mask = reference_values.isin([0, 1])
    valid_prediction_mask = prediction_values.isin([0, 1])
    evaluation_mask = reference_mask & valid_prediction_mask
    evaluated = frame.loc[evaluation_mask].copy()
    evaluated["model_answer"] = prediction_values.loc[evaluation_mask].astype(int)
    evaluated[reference_column] = reference_values.loc[evaluation_mask].astype(int)

    if evaluated.empty:
        raise ValueError("no rows with both prediction and reference")

    attempted_total = int(reference_mask.sum())
    valid_total = len(evaluated)
    invalid_predictions = attempted_total - valid_total
    predictions = evaluated["model_answer"]
    references = evaluated[reference_column]
    return {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(
            references, predictions, pos_label=1, zero_division=0
        ),
        "recall": recall_score(
            references, predictions, pos_label=1, zero_division=0
        ),
        "f1": f1_score(
            references, predictions, pos_label=1, zero_division=0
        ),
        "f1_macro": f1_score(
            references, predictions, average="macro", zero_division=0
        ),
        "total": valid_total,
        "attempted_total": attempted_total,
        "invalid_predictions": invalid_predictions,
        "prediction_coverage": valid_total / attempted_total,
        "correct": int((predictions == references).sum()),
        "label_0": int((references == 0).sum()),
        "label_1": int((references == 1).sum()),
        "pred_0": int((predictions == 0).sum()),
        "pred_1": int((predictions == 1).sum()),
    }


def discover_and_evaluate() -> pd.DataFrame:
    rows = []
    skipped = []
    evaluations = []
    block_members = {}

    prediction_roots = [
        OUTPUT_DIR / "decoders",
        OUTPUT_DIR / "encoders",
    ]
    prediction_files = [
        file_path
        for root in prediction_roots
        if root.exists()
        for file_path in root.rglob("*.csv")
    ]

    for file_path in sorted(prediction_files):
        if not any(input_name in file_path.stem for input_name in INPUT_TYPES):
            continue

        try:
            frame = pd.read_csv(readable_path(file_path))
            if "model_answer" not in frame.columns:
                continue
            relative_parts = file_path.relative_to(OUTPUT_DIR).parts
            if (
                relative_parts[0] == "encoders"
                and ENCODER_SPLIT_PROTOCOL not in relative_parts
            ):
                continue
            info = parse_prediction_info(file_path)
            if info["model"] not in MODEL_LABELS:
                continue
            reference_specs = (
                [("Human majority vote", "human_majority_label")]
                if info["cohort"] == "Human held-out"
                else [("Dataset ground truth", "label")]
            )

            for reference_name, reference_column in reference_specs:
                valid_ids = valid_evaluation_ids(frame, reference_column)
                if not valid_ids:
                    raise ValueError(
                        "no rows with both prediction and reference"
                    )
                block_key = (
                    info["cohort"],
                    reference_name,
                    info["dataset_label"],
                )
                evaluations.append(
                    {
                        "frame": frame,
                        "info": info,
                        "reference": reference_name,
                        "reference_column": reference_column,
                        "file_path": file_path,
                        "block_key": block_key,
                    }
                )
                block_members.setdefault(block_key, []).append(valid_ids)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            skipped.append((file_path, str(error)))

    if skipped:
        print("\nSkipped prediction files:")
        for file_path, reason in skipped:
            print(f"  - {file_path}: {reason}")

    common_ids_by_block = {}
    print("\nCommon valid samples per evaluation block:")
    for block_key, valid_id_sets in sorted(block_members.items()):
        common_ids = set.intersection(*valid_id_sets)
        if not common_ids:
            raise ValueError(
                f"no common valid predictions for block {block_key}"
            )
        common_ids_by_block[block_key] = common_ids
        cohort, reference, dataset = block_key
        print(
            f"  - {cohort} | {reference} | {dataset}: "
            f"{len(common_ids)} samples"
        )

    for evaluation in evaluations:
        frame = evaluation["frame"]
        common_ids = common_ids_by_block[evaluation["block_key"]]
        normalized_ids = frame["unique_id"].astype("string")
        common_frame = frame.loc[normalized_ids.isin(common_ids)].copy()
        metrics = compute_metrics(
            common_frame,
            evaluation["reference_column"],
        )
        rows.append(
            {
                **evaluation["info"],
                "reference": evaluation["reference"],
                **metrics,
                "file_path": str(evaluation["file_path"]),
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No valid prediction CSV files found under {OUTPUT_DIR}"
        )
    return pd.DataFrame(rows)


def generate_summary_csv(results: pd.DataFrame) -> pd.DataFrame:
    summary_source = results.copy()
    summary_source["model_technique"] = (
        summary_source["model_label"]
        + "_"
        + summary_source["technique_short"]
    )
    summary = summary_source.pivot_table(
        index=["cohort", "reference", "dataset_label", "input_type"],
        columns="model_technique",
        values="f1",
        aggfunc="first",
    ).reset_index()
    summary.columns.name = None
    summary = summary.rename(
        columns={
            "cohort": "Cohort",
            "reference": "Reference",
            "dataset_label": "Dataset",
            "input_type": "Input",
        }
    )
    summary.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)

    macro_summary = summary_source.pivot_table(
        index=["cohort", "reference", "dataset_label", "input_type"],
        columns="model_technique",
        values="f1_macro",
        aggfunc="first",
    ).reset_index()
    macro_summary.columns.name = None
    macro_summary = macro_summary.rename(
        columns={
            "cohort": "Cohort",
            "reference": "Reference",
            "dataset_label": "Dataset",
            "input_type": "Input",
        }
    )
    macro_summary.to_csv(
        OUTPUT_DIR / "summary_results_macro_f1.csv",
        index=False,
    )
    return summary


def print_console_results(results: pd.DataFrame) -> None:
    display_columns = [
        "cohort",
        "reference",
        "dataset_label",
        "model_label",
        "technique",
        "input_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "total",
    ]
    display = results[display_columns].sort_values(
        [
            "cohort",
            "reference",
            "dataset_label",
            "model_label",
            "technique",
            "input_type",
        ]
    )
    print("\n" + display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    results = discover_and_evaluate()
    results = results.sort_values(
        [
            "cohort",
            "reference",
            "dataset_label",
            "model_label",
            "technique",
            "input_type",
        ]
    )
    results.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    summary = generate_summary_csv(results)
    print_console_results(results)
    print(f"\nDetailed results: {OUTPUT_DIR / 'evaluation_results.csv'}")
    print(f"F1 summary: {OUTPUT_DIR / 'summary_results.csv'}")
    print(
        "Macro-F1 summary: "
        f"{OUTPUT_DIR / 'summary_results_macro_f1.csv'}"
    )
    print(f"Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
