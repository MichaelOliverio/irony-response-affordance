import pandas as pd
from evaluate import load
from statsmodels.stats.contingency_tables import mcnemar
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

# =========================================================
def compute_metrics(df, target_col, accuracy_metric, precision_metric, recall_metric, f1_metric):
    df_eval = df[df[target_col].notna()].copy()
    if len(df_eval) == 0:
        return None
    
    # Reset index per evitare problemi con indici non consecutivi
    df_eval = df_eval.reset_index(drop=True)

    df_eval["model_answer"] = df_eval["model_answer"].astype(int)
    df_eval[target_col] = df_eval[target_col].astype(int)

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
        "n_samples": len(df_eval),
    }

def mcnemar_test(df, label_col, semi_col):
    df_eval = df[df[semi_col].notna()].copy()
    if len(df_eval) == 0:
        return None

    correct_label = df_eval["model_answer"] == df_eval[label_col]
    correct_semi = df_eval["model_answer"] == df_eval[semi_col]

    b = ((correct_label == True) & (correct_semi == False)).sum()
    c = ((correct_label == False) & (correct_semi == True)).sum()

    table = [[0, b],
             [c, 0]]

    result = mcnemar(table, exact=(b + c < 25))

    return {
        "mcnemar_p_value": result.pvalue,
        "mcnemar_b": b,
        "mcnemar_c": c,
    }

# =========================================================
def main():
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    results = []

    files = {
        'MultiPICo': {
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/gender_female_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/gender_male_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_boomer_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_x_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_y_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_z_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
            ],
            'mistralai_Ministral-8B-Instruct-2410': [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/gender_female_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/gender_male_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_boomer_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_x_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_y_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_z_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
            ],
            'google_gemma-2-9b-it': [
                "./output/decoders/google_gemma-2-9b-it/gender_female_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/gender_male_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_boomer_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_x_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_y_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_z_google_gemma-2-9b-it_MultiPICo_conversation.csv",
            ]
        },
        'CSC': {
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/gender_female_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/gender_male_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_boomer_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_x_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_y_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/generation_gen_z_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/relationship_colleagues_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/relationship_friends_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
            ],
            'mistralai_Ministral-8B-Instruct-2410': [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/gender_female_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/gender_male_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_boomer_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_x_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_y_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/generation_gen_z_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/relationship_colleagues_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/relationship_friends_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
            ],
            'google_gemma-2-9b-it': [
                "./output/decoders/google_gemma-2-9b-it/gender_female_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/gender_male_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_boomer_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_x_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_y_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/generation_gen_z_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/relationship_colleagues_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/relationship_friends_google_gemma-2-9b-it_CSC_conversation.csv",
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
                df["model_answer"] = df["model_answer"].astype(int)

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
                        "mcnemar_p_value": None,
                        "mcnemar_b": None,
                        "mcnemar_c": None,
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

                mc = mcnemar_test(df, "label", semi_col)

                results.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "file": file_path,
                    "strategy": strategy,
                    "target_label": semi_col,
                    **res_semi,
                    "n_discarded_not_in_dataset": n_discarded_not_in_dataset,
                    "mcnemar_p_value": None if mc is None else mc["mcnemar_p_value"],
                    "mcnemar_b": None if mc is None else mc["mcnemar_b"],
                    "mcnemar_c": None if mc is None else mc["mcnemar_c"],
                })

    # =====================================================
    # SALVATAGGIO
    # =====================================================
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