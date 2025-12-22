import pandas as pd
from evaluate import load
from src.config import DATASETS
import json 

# Tutte le possibili ground truth labels disponibili nei dataset
ALL_GROUND_TRUTHS = [
    "aggregate_sarcasm_label",  # per CSC
    "aggregate_irony_label",    # per MultiPICo
    "aggregation_on_male",
    "aggregation_on_female",
    "aggregation_on_boomer",
    "aggregation_on_genz",
    "aggregation_on_geny",
    "aggregation_on_genx",
    "aggregation_on_colleague",
    "aggregation_on_best_friend",
]

OUTPUT_CSV = "./evaluation_results_llm_perspectives.csv"
OUTPUT_JSON = "./evaluation_results_llm_perspectives.json"

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
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/no_perspective_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
            ],
            'mistralai_Ministral-8B-Instruct-2410': [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/no_perspective_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
            ],
            'google_gemma-2-9b-it': [
                "./output/decoders/google_gemma-2-9b-it/no_perspective_google_gemma-2-9b-it_MultiPICo_conversation.csv",
            ]
        },
        'CSC': {
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/no_perspective_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
            ],
            'mistralai_Ministral-8B-Instruct-2410': [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/no_perspective_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
            ],
            'google_gemma-2-9b-it': [
                "./output/decoders/google_gemma-2-9b-it/no_perspective_google_gemma-2-9b-it_CSC_conversation.csv",
            ],
        },
    }

    for dataset_name, models in files.items():
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Caricamento del dataset originale
        dataset_path = DATASETS[dataset_name]['path']
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = data["entries"]

        # Determina la label aggregata principale per questo dataset
        main_label_key = "aggregate_sarcasm_label" if dataset_name == 'CSC' else "aggregate_irony_label"
        
        # Creiamo un dizionario per tutte le label dal dataset originale
        # E tracciamo quali unique_id hanno voti validi per ogni ground truth
        labels_dict = {}
        valid_ids_per_gt = {gt: set() for gt in ALL_GROUND_TRUTHS}
        valid_ids_per_gt["main_label"] = set()
        
        for entry in dataset:
            unique_id = entry.get("unique_id")
            annotations = entry.get("annotations", {})
            
            labels_dict[unique_id] = {}
            
            # Aggiungi la label principale
            main_label_value = annotations.get(main_label_key)
            labels_dict[unique_id]["main_label"] = main_label_value
            if main_label_value is not None:
                valid_ids_per_gt["main_label"].add(unique_id)
            
            # Aggiungi tutte le altre ground truth disponibili
            for gt_label in ALL_GROUND_TRUTHS:
                if gt_label == main_label_key:
                    continue  # Già aggiunta come main_label
                if gt_label in annotations:
                    gt_value = annotations.get(gt_label)
                    labels_dict[unique_id][gt_label] = gt_value
                    
                    # Verifica se ci sono voti validi per questa ground truth
                    vote_key = f"{gt_label}_n"
                    if vote_key in annotations and gt_value is not None:
                        valid_ids_per_gt[gt_label].add(unique_id)
                    
        for model_name, file_paths in models.items():
            print(f"  Processing model: {model_name}")
            
            for file_path in file_paths:
                print(f"    Processing file: {file_path}")
                
                # Carichiamo il file con le predizioni del modello
                df = pd.read_csv(file_path)
                
                # Aggiungiamo tutte le label disponibili dal dataset originale
                for col_name in ["main_label"] + ALL_GROUND_TRUTHS:
                    df[col_name] = df["unique_id"].map(
                        lambda uid: labels_dict.get(uid, {}).get(col_name)
                    )
                
                # Reset index per evitare problemi
                df = df.reset_index(drop=True)
                df["model_answer"] = df["model_answer"].astype(int)
                
                # Valuta su tutte le ground truth disponibili
                # 1) Main label (aggregated)
                df_main = df[df["unique_id"].isin(valid_ids_per_gt["main_label"])].copy()
                
                if len(df_main) > 0:
                    res_main = compute_metrics(
                        df_main, "main_label",
                        accuracy_metric, precision_metric, recall_metric, f1_metric
                    )
                    
                    if res_main is not None:
                        results.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "file": file_path,
                            "ground_truth": main_label_key,
                            **res_main,
                        })
                        print(f"      Evaluated on {main_label_key}: {res_main['n_samples']} samples")
                
                # 2) Tutte le altre ground truth
                for gt_label in ALL_GROUND_TRUTHS:
                    if gt_label == main_label_key:
                        continue  # Già valutata
                    
                    # Salta relationship per MultiPICo
                    if dataset_name == 'MultiPICo' and gt_label in ['aggregation_on_colleague', 'aggregation_on_best_friend']:
                        continue
                    
                    # Salta se non ci sono unique_id validi per questa ground truth
                    if gt_label not in valid_ids_per_gt or len(valid_ids_per_gt[gt_label]) == 0:
                        continue
                    
                    # Filtra solo i campioni con voti validi per questa ground truth
                    df_gt = df[df["unique_id"].isin(valid_ids_per_gt[gt_label])].copy()
                    
                    if len(df_gt) == 0:
                        continue
                    
                    res_gt = compute_metrics(
                        df_gt, gt_label,
                        accuracy_metric, precision_metric, recall_metric, f1_metric
                    )
                    
                    if res_gt is not None:
                        results.append({
                            "dataset": dataset_name,
                            "model": model_name,
                            "file": file_path,
                            "ground_truth": gt_label,
                            **res_gt,
                        })
                        print(f"      Evaluated on {gt_label}: {res_gt['n_samples']} samples")

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

    print(f"\n✓ Results saved to: {OUTPUT_CSV}")
    print(f"✓ Results saved to: {OUTPUT_JSON}")
    print(f"✓ Total rows: {len(df_results)}")

# =========================================================
if __name__ == "__main__":
    main()