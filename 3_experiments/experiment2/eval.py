import pandas as pd
import numpy as np
from evaluate import load
from statsmodels.stats.multitest import multipletests
from src.config import DATASETS
import json
from pathlib import Path

# Directory di questo file
_HERE = Path(__file__).resolve().parent

# File sorgente da cui leggere gli unique_id validi (entries filtrate dal LLM judge)
VALID_IDS_FILES = {
    'MultiPICo': _HERE.parent / "datasets" / "MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_final.json",
    'CSC': _HERE.parent / "datasets" / "CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_final.json",
}

STRATEGY_TO_TARGET = {
    "gender_male": "aggregation_on_male",
    "gender_female": "aggregation_on_female",
    "generation_boomer": "aggregation_on_boomer",
    "generation_gen_z": "aggregation_on_genz",
    "generation_gen_y": "aggregation_on_geny",
    "generation_gen_x": "aggregation_on_genx",
    "relationship_colleagues": "aggregation_on_colleague",
    "relationship_friends": "aggregation_on_best_friend",
}

OUTPUT_CSV = "./evaluation_results_semiaggregated.csv"
OUTPUT_JSON = "./evaluation_results_semiaggregated.json"

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_SEED = 42

# =========================================================
def compute_metrics(df, target_col, accuracy_metric, precision_metric, recall_metric, f1_metric):
    prediction_values = pd.to_numeric(df["model_answer"], errors="coerce")
    target_values = pd.to_numeric(df[target_col], errors="coerce")
    target_mask = target_values.isin([0, 1])
    valid_prediction_mask = prediction_values.isin([0, 1])
    evaluation_mask = target_mask & valid_prediction_mask
    df_eval = df.loc[evaluation_mask].copy()
    if len(df_eval) == 0:
        return None

    valid_predictions = (
        prediction_values.loc[evaluation_mask].astype(int).to_numpy()
    )
    valid_targets = target_values.loc[evaluation_mask].astype(int).to_numpy()

    # Reset index per evitare problemi con indici non consecutivi
    df_eval = df_eval.reset_index(drop=True)

    df_eval["model_answer"] = valid_predictions
    df_eval[target_col] = valid_targets
    n_attempted = int(target_mask.sum())
    n_samples = len(df_eval)

    return {
        "accuracy": accuracy_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col]
        )["accuracy"],
        "precision": precision_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["precision"],
        "recall": recall_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["recall"],
        "f1": f1_metric.compute(
            predictions=df_eval["model_answer"],
            references=df_eval[target_col],
            pos_label=1,
            average="binary"
        )["f1"],
        "n_samples": n_samples,
        "n_attempted": n_attempted,
        "n_invalid_predictions": n_attempted - n_samples,
        "prediction_coverage": n_samples / n_attempted,
    }

def _positive_class_f1(references, predictions):
    """Compute binary F1 for label 1, with zero_division=0 semantics."""
    true_positive = np.count_nonzero((references == 1) & (predictions == 1))
    false_positive = np.count_nonzero((references == 0) & (predictions == 1))
    false_negative = np.count_nonzero((references == 1) & (predictions == 0))
    denominator = (2 * true_positive) + false_positive + false_negative
    return 0.0 if denominator == 0 else (2 * true_positive) / denominator


def paired_bootstrap_f1_test(
    df,
    label_col,
    semi_col,
    n_iterations=BOOTSTRAP_ITERATIONS,
    alpha=BOOTSTRAP_ALPHA,
    random_seed=BOOTSTRAP_SEED,
):
    """Compare positive-class F1 against two references on paired items."""
    prediction_values = pd.to_numeric(df["model_answer"], errors="coerce")
    aggregated_values = pd.to_numeric(df[label_col], errors="coerce")
    perspective_values = pd.to_numeric(df[semi_col], errors="coerce")
    evaluation_mask = (
        prediction_values.isin([0, 1])
        & aggregated_values.isin([0, 1])
        & perspective_values.isin([0, 1])
    )
    df_eval = df.loc[evaluation_mask].copy()
    if len(df_eval) == 0:
        return None

    predictions = prediction_values.loc[evaluation_mask].astype(int).to_numpy()
    aggregated = aggregated_values.loc[evaluation_mask].astype(int).to_numpy()
    perspective = perspective_values.loc[evaluation_mask].astype(int).to_numpy()
    n_samples = len(df_eval)

    aggregated_f1 = _positive_class_f1(aggregated, predictions)
    perspective_f1 = _positive_class_f1(perspective, predictions)

    rng = np.random.default_rng(random_seed)
    bootstrap_deltas = np.empty(n_iterations, dtype=float)
    for iteration in range(n_iterations):
        sampled_indices = rng.integers(0, n_samples, size=n_samples)
        sampled_predictions = predictions[sampled_indices]
        bootstrap_deltas[iteration] = (
            _positive_class_f1(
                perspective[sampled_indices], sampled_predictions
            )
            - _positive_class_f1(
                aggregated[sampled_indices], sampled_predictions
            )
        )

    ci_low, ci_high = np.quantile(
        bootstrap_deltas, [alpha / 2, 1 - (alpha / 2)]
    )
    lower_tail = (
        np.count_nonzero(bootstrap_deltas <= 0) + 1
    ) / (n_iterations + 1)
    upper_tail = (
        np.count_nonzero(bootstrap_deltas >= 0) + 1
    ) / (n_iterations + 1)
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))

    return {
        "f1_aggregated": aggregated_f1,
        "f1_perspective": perspective_f1,
        "f1_delta": perspective_f1 - aggregated_f1,
        "f1_bootstrap_ci_low": float(ci_low),
        "f1_bootstrap_ci_high": float(ci_high),
        "f1_bootstrap_p_value": float(p_value),
        "f1_bootstrap_significant_uncorrected": bool(
            ci_low > 0 or ci_high < 0
        ),
        "bootstrap_iterations": n_iterations,
        "bootstrap_seed": random_seed,
    }


def apply_holm_correction(results, alpha=BOOTSTRAP_ALPHA):
    """Correct bootstrap p-values within each dataset-model family."""
    families = {}
    for index, result in enumerate(results):
        p_value = result.get("f1_bootstrap_p_value")
        if p_value is None:
            continue
        family = (result["dataset"], result["model"])
        families.setdefault(family, []).append(index)

    for result_indices in families.values():
        raw_p_values = [
            results[index]["f1_bootstrap_p_value"]
            for index in result_indices
        ]
        rejected, adjusted_p_values, _, _ = multipletests(
            raw_p_values,
            alpha=alpha,
            method="holm",
        )
        family_size = len(result_indices)
        for index, adjusted_p, is_significant in zip(
            result_indices, adjusted_p_values, rejected
        ):
            results[index]["f1_bootstrap_p_value_holm"] = float(
                adjusted_p
            )
            results[index]["f1_bootstrap_significant_holm"] = bool(
                is_significant
            )
            results[index]["holm_family_size"] = family_size

    return results

# =========================================================
def main():
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    results = []

    files = {
        'MultiPICo': {
            "Qwen_Qwen3.5-9B": [
                "./output/decoders/Qwen_Qwen3.5-9B/gender_female_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/gender_male_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_boomer_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_x_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_y_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_z_Qwen_Qwen3.5-9B_MultiPICo_conversation.csv",
            ],
            'mistralai_Ministral-3-8B-Instruct-2512': [
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/gender_female_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/gender_male_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_boomer_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_x_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_y_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_z_mistralai_Ministral-3-8B-Instruct-2512_MultiPICo_conversation.csv",
            ],
            'google_gemma-4-E4B-it': [
                "./output/decoders/google_gemma-4-E4B-it/gender_female_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/gender_male_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_boomer_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_x_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_y_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_z_google_gemma-4-E4B-it_MultiPICo_conversation.csv",
            ]
        },
        'CSC': {
            "Qwen_Qwen3.5-9B": [
                "./output/decoders/Qwen_Qwen3.5-9B/gender_female_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/gender_male_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_boomer_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_x_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_y_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/generation_gen_z_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/relationship_colleagues_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/relationship_friends_Qwen_Qwen3.5-9B_CSC_conversation.csv",
            ],
            #'mistralai_Ministral-3-8B-Instruct-2512': [
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/gender_female_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/gender_male_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_boomer_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_x_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_y_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/generation_gen_z_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/relationship_colleagues_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #    "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/relationship_friends_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            #],
            'google_gemma-4-E4B-it': [
                "./output/decoders/google_gemma-4-E4B-it/gender_female_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/gender_male_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_boomer_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_x_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_y_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/generation_gen_z_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/relationship_colleagues_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/relationship_friends_google_gemma-4-E4B-it_CSC_conversation.csv",
            ],
        },
    }


    for dataset_name, models in files.items():                  
        for model_name, file_paths in models.items():
            for file_path in file_paths:
                # Prendiamo la strategia di aggregazione dal nome del file
                strategy = None
                for s in STRATEGY_TO_TARGET:
                    if s in file_path:
                        strategy = s
                        break
                
                if strategy is None:
                    raise ValueError(f"Strategy not found in file path: {file_path}")
                semi_col = STRATEGY_TO_TARGET[strategy]

                # Caricamento degli unique_id validi dai file _final.json
                valid_ids_path = VALID_IDS_FILES[dataset_name]
                with open(valid_ids_path, 'r', encoding='utf-8') as f:
                    valid_ids_data = json.load(f)
                dataset_unique_ids = {entry["unique_id"] for entry in valid_ids_data["entries"]}

                # Caricamento del dataset originale (con annotazioni semi-aggregate)
                dataset_path = DATASETS[dataset_name]['path']
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                dataset = data["entries"]

                # Creiamo un dizionario per le label dal dataset originale
                labels_dict = {}
                unique_ids_to_exclude = set()

                for entry in dataset:
                    unique_id = entry.get("unique_id")
                    annotations = entry.get("annotations", {})
                    
                    label_key = None
                    if dataset_name == 'CSC':
                        label_key = "aggregate_sarcasm_label"
                    else:
                        label_key = "aggregate_irony_label"

                    # Salviamo la label aggregata
                    labels_dict[unique_id] = {
                        "label": annotations.get(label_key),
                        semi_col: annotations.get(semi_col)
                    }

                    # Escludiamo gli esempi senza voti per la strategia corrente
                    exclude_entry = False

                    if (not dataset_name == 'CSC') and (semi_col in ['aggregation_on_colleague', 'aggregation_on_best_friend']):
                        continue

                    vote_key = f"{semi_col}_n"
                    if vote_key not in annotations:
                        exclude_entry = True

                    if exclude_entry:
                        unique_ids_to_exclude.add(unique_id)

                print(f"{semi_col}: Excluding {len(unique_ids_to_exclude)} unique IDs for dataset {dataset_name}.")

                # Carichiamo il file con le predizioni del modello
                df = pd.read_csv(file_path)
                n_before_dataset_filter = len(df)

                # Filtriamo tenendo solo gli unique_id presenti nel dataset
                df = df[df["unique_id"].isin(dataset_unique_ids)].copy()
                n_discarded_not_in_dataset = n_before_dataset_filter - len(df)
                print(f"  [{file_path}] Record nel CSV: {n_before_dataset_filter}, "
                      f"scartati (non nel dataset): {n_discarded_not_in_dataset}, "
                      f"rimasti: {len(df)}")

                # Aggiungiamo le label dal dataset originale
                df["label"] = df["unique_id"].map(lambda uid: labels_dict.get(uid, {}).get("label"))
                df[semi_col] = df["unique_id"].map(lambda uid: labels_dict.get(uid, {}).get(semi_col))
                
                # Escludiamo gli esempi senza voti
                df = df[~df["unique_id"].isin(unique_ids_to_exclude)].copy()
                
                # Verifica se il DataFrame è vuoto dopo l'esclusione
                if len(df) == 0:
                    print(f"Skipping {file_path}: no samples left after exclusion.")
                    continue
                
                # Reset index per evitare problemi
                df = df.reset_index(drop=True)
                # -----------------------------
                # 1) Aggregated label
                # -----------------------------
                res_label = compute_metrics(
                    df, "label",
                    accuracy_metric, precision_metric, recall_metric, f1_metric
                )

                if res_label is not None:
                    results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "file": file_path,
                        "strategy": "aggregated",
                        "target_label": "label",
                        **res_label,
                        "n_discarded_not_in_dataset": n_discarded_not_in_dataset,
                    })

                # -----------------------------
                # 2) Semi-aggregated (se strategy nel filename)
                # -----------------------------
                strategy = None
                for s in STRATEGY_TO_TARGET:
                    if s in file_path:
                        strategy = s
                        break

                if strategy is None:
                    continue

                semi_col = STRATEGY_TO_TARGET[strategy]
                if semi_col not in df.columns:
                    continue

                res_semi = compute_metrics(
                    df, semi_col,
                    accuracy_metric, precision_metric, recall_metric, f1_metric
                )

                if res_semi is None:
                    continue

                bootstrap = paired_bootstrap_f1_test(
                    df, "label", semi_col
                )
                if bootstrap is None:
                    continue

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "file": file_path,
                    "strategy": strategy,
                    "target_label": semi_col,
                    **res_semi,
                    **bootstrap,
                    "n_discarded_not_in_dataset": n_discarded_not_in_dataset,
                })

    # =====================================================
    # SALVATAGGIO
    # =====================================================
    results = apply_holm_correction(results)
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    # Salvataggio JSON - converti i tipi numpy in tipi Python standard
    results_json = []
    for result in results:
        result_dict = {}
        for key, value in result.items():
            if pd.isna(value):
                result_dict[key] = None
            elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)) or hasattr(value, 'item'):
                result_dict[key] = value.item() if hasattr(value, 'item') else value
            elif isinstance(value, (int, float, str, bool, type(None))):
                result_dict[key] = value
            else:
                result_dict[key] = str(value)
        results_json.append(result_dict)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to: {OUTPUT_CSV}")
    print(f" Results saved to: {OUTPUT_JSON}")
    print(f"Total rows: {len(df_results)}")

# =========================================================
if __name__ == "__main__":
    main()
