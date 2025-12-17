import pandas as pd
from evaluate import load
from statsmodels.stats.contingency_tables import mcnemar
from src.config import DATASETS
import json 

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

                # Escludiamo gli esempi senza voti per la strategia corrente
                unique_ids_to_exclude = set()
                dataset_path = DATASETS[dataset_name]['path']
                
                # Caricamento del dataset per ottenere gli ID unici da escludere
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                dataset = data["entries"]

                for entry in dataset:
                    annotations = entry.get("annotations", {})
                    exclude_entry = False

                    if (not dataset_name == 'CSC') and (semi_col in ['aggregation_on_colleague', 'aggregation_on_best_friend']):
                        continue

                    vote_key = f"{semi_col}_n"
                    if vote_key not in annotations:
                        #print(f"Excluding entry with unique_id {entry.get('unique_id')} due to missing vote_key: {vote_key}")
                        exclude_entry = True

                    if exclude_entry:
                        unique_ids_to_exclude.add(entry.get("unique_id"))

                
                print(f"{semi_col}: Excluding {len(unique_ids_to_exclude)} unique IDs for dataset {dataset_name}.")

                df = pd.read_csv(file_path)
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