"""Evaluate Experiment 1 predictions against dataset and human references."""

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
    #"mistralai_Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
    #"google_gemma-4-E4B-it": "Gemma-4-E4B",
    #"FacebookAI_xlm-roberta-large": "XLM-RoBERTa-large",
    #"EuroBERT_EuroBERT-210m": "EuroBERT-210M",
    #"EuroBERT_EuroBERT-610m": "EuroBERT-610M",
}

INPUT_TYPES = (
    "context_reply",
    "last_u1_message",
    "conversation",
)


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


def compute_metrics(frame: pd.DataFrame, reference_column: str) -> dict:
    required_columns = {"model_answer", reference_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing columns: {sorted(missing_columns)}")

    evaluated = frame.dropna(subset=list(required_columns)).copy()
    evaluated["model_answer"] = evaluated["model_answer"].astype(int)
    evaluated[reference_column] = evaluated[reference_column].astype(int)

    if evaluated.empty:
        raise ValueError("no rows with both prediction and reference")

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
        "total": len(evaluated),
        "correct": int((predictions == references).sum()),
        "label_0": int((references == 0).sum()),
        "label_1": int((references == 1).sum()),
        "pred_0": int((predictions == 0).sum()),
        "pred_1": int((predictions == 1).sum()),
    }


def discover_and_evaluate() -> pd.DataFrame:
    rows = []
    skipped = []

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
        try:
            frame = pd.read_csv(file_path)
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
                metrics = compute_metrics(frame, reference_column)
                rows.append(
                    {
                        **info,
                        "reference": reference_name,
                        **metrics,
                        "file_path": str(file_path),
                    }
                )
        except (OSError, ValueError, pd.errors.ParserError) as error:
            skipped.append((file_path, str(error)))

    if skipped:
        print("\nSkipped prediction files:")
        for file_path, reason in skipped:
            print(f"  - {file_path}: {reason}")

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
