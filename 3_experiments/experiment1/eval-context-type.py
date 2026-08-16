import pandas as pd
import json
from evaluate import load
from datetime import datetime
from pathlib import Path


def load_context_type_mapping(dataset_name):
    """Carica il mapping unique_id -> context_type dai dataset originali."""
    base_path = Path("../datasets")
    
    if dataset_name == "CSC":
        file_path = base_path / "CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered.json"
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")
    
    print(f"Loading context_type mapping from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data['entries']

    # Crea mapping unique_id -> context_type
    mapping = {}
    for entry in dataset:
        unique_id = entry.get("unique_id")
        context_type = entry.get("context_type")
        if unique_id and context_type:
            mapping[unique_id] = context_type
    
    print(f"✓ Loaded {len(mapping)} context_type mappings for {dataset_name}")
    return mapping


def parse_file_info(file_path, dataset_name):
    """Estrae informazioni dal nome del file."""
    input_type = file_path.split(f"_{dataset_name}_")[1].split(".csv")[0]
    
    if "zero_shot" in file_path:
        prompting_method = "Zero-Shot"
    elif "chain_of_thought" in file_path:
        prompting_method = "Chain-of-Thought"
    elif "fine_tuning" in file_path:
        prompting_method = "Fine-Tuning"
    else:
        prompting_method = "Unknown"
    
    return input_type, prompting_method


def compute_metrics(df, accuracy_metric, precision_metric, recall_metric, f1_metric):
    """Calcola tutte le metriche per un dataset."""
    df["model_answer"] = df["model_answer"].astype(int)
    
    # Usa context_type invece di label
    accuracy = accuracy_metric.compute(
        predictions=df["model_answer"], 
        references=df["context_type"]
    )["accuracy"]
    
    precision = precision_metric.compute(
        predictions=df["model_answer"], 
        references=df["context_type"], 
        pos_label=1,
        average="binary"
    )["precision"]
    
    recall = recall_metric.compute(
        predictions=df["model_answer"], 
        references=df["context_type"], 
        pos_label=1,
        average="binary"
    )["recall"]
    
    f1 = f1_metric.compute(
        predictions=df["model_answer"], 
        references=df["context_type"], 
        pos_label=1,
        average="binary"
    )["f1"]
    
    # Statistiche aggiuntive
    total = len(df)
    correct = (df["model_answer"] == df["context_type"]).sum()
    label_0 = (df["context_type"] == 0).sum()
    label_1 = (df["context_type"] == 1).sum()
    pred_0 = (df["model_answer"] == 0).sum()
    pred_1 = (df["model_answer"] == 1).sum()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "label_0": label_0,
        "label_1": label_1,
        "pred_0": pred_0,
        "pred_1": pred_1
    }


def print_console_results(results_data):
    """Stampa i risultati in console in formato leggibile."""
    for dataset_name, dataset_results in results_data.items():
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}".center(100))
        print("=" * 100)
        
        for model_name, model_results in dataset_results.items():
            print(f"\n{'MODEL: ' + model_name}".center(100, '-'))
            
            # Stampa header della tabella
            print(f"\n{'Technique':<20} {'Input Type':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Samples':<10}")
            print("-" * 100)
            
            for result in model_results:
                print(f"{result['technique']:<20} {result['input_type']:<20} "
                      f"{result['metrics']['accuracy']:<8.4f} "
                      f"{result['metrics']['precision']:<8.4f} "
                      f"{result['metrics']['recall']:<8.4f} "
                      f"{result['metrics']['f1']:<8.4f} "
                      f"{result['metrics']['total']:<10}")

def main():
    # Carica le metriche da Hugging Face Evaluate
    print("Loading evaluation metrics...")
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    # File da valutare
    files = {
        'CSC': {
            "EuroBERT_EuroBERT-210m": [
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/context_reply/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_context_reply.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/last_u1_message/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_last_u1_message.csv",
                #"./output/encoders/EuroBERT_EuroBERT-210m/CSC/context/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_context.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/conversation/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_conversation.csv",
            ],
            "FacebookAI_xlm-roberta-large": [
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/context/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_context.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/last_u1_message/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_last_u1_message.csv",
                #"./output/encoders/FacebookAI_xlm-roberta-large/CSC/context_reply/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_context_reply.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/conversation/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_conversation.csv",
            ],
            "google_gemma-4-E4B-it": [
                "./output/decoders/google_gemma-4-E4B-it/zero_shot_google_gemma-4-E4B-it_CSC_context_reply.csv",
                "./output/decoders/google_gemma-4-E4B-it/zero_shot_google_gemma-4-E4B-it_CSC_last_u1_message.csv",
                #"./output/decoders/google_gemma-4-E4B-it/zero_shot_google_gemma-4-E4B-it_CSC_context.csv",
                "./output/decoders/google_gemma-4-E4B-it/zero_shot_google_gemma-4-E4B-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-4-E4B-it/chain_of_thought_google_gemma-4-E4B-it_CSC_context_reply.csv",
                "./output/decoders/google_gemma-4-E4B-it/chain_of_thought_google_gemma-4-E4B-it_CSC_last_u1_message.csv",
                #"./output/decoders/google_gemma-4-E4B-it/chain_of_thought_google_gemma-4-E4B-it_CSC_context.csv",
                "./output/decoders/google_gemma-4-E4B-it/chain_of_thought_google_gemma-4-E4B-it_CSC_conversation.csv",
            ],
            "Qwen_Qwen3.5-9B": [
                "./output/decoders/Qwen_Qwen3.5-9B/zero_shot_Qwen_Qwen3.5-9B_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/zero_shot_Qwen_Qwen3.5-9B_CSC_last_u1_message.csv",
                #"./output/decoders/Qwen_Qwen3.5-9B/zero_shot_Qwen_Qwen3.5-9B_CSC_context.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/zero_shot_Qwen_Qwen3.5-9B_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/chain_of_thought_Qwen_Qwen3.5-9B_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/chain_of_thought_Qwen_Qwen3.5-9B_CSC_last_u1_message.csv",
                #"./output/decoders/Qwen_Qwen3.5-9B/chain_of_thought_Qwen_Qwen3.5-9B_CSC_context.csv",
                "./output/decoders/Qwen_Qwen3.5-9B/chain_of_thought_Qwen_Qwen3.5-9B_CSC_conversation.csv",
            ],
            "mistralai_Ministral-3-8B-Instruct-2512": [
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/zero_shot_mistralai_Ministral-3-8B-Instruct-2512_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/zero_shot_mistralai_Ministral-3-8B-Instruct-2512_CSC_last_u1_message.csv",
                #"./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/zero_shot_mistralai_Ministral-3-8B-Instruct-2512_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/zero_shot_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/chain_of_thought_mistralai_Ministral-3-8B-Instruct-2512_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/chain_of_thought_mistralai_Ministral-3-8B-Instruct-2512_CSC_last_u1_message.csv",
                #"./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/chain_of_thought_mistralai_Ministral-3-8B-Instruct-2512_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-3-8B-Instruct-2512/chain_of_thought_mistralai_Ministral-3-8B-Instruct-2512_CSC_conversation.csv",
            ],
        },
    }

    # Struttura per memorizzare tutti i risultati
    results_data = {}
    
    # Carica i mapping context_type per entrambi i dataset
    context_type_mappings = {
        'CSC': load_context_type_mapping('CSC'),
    }
    
    print("\nEvaluating models...\n")

    for dataset_name, dataset_files in files.items():
        if dataset_name == 'MultiPICo':
            break

        results_data[dataset_name] = {}
        context_mapping = context_type_mappings[dataset_name]
        
        for model_name, file_paths in dataset_files.items():
            results_data[dataset_name][model_name] = []
            
            for file_path in file_paths:
                # Parsing delle informazioni dal file
                input_type, prompting_method = parse_file_info(file_path, dataset_name)
                
                # Carica il CSV
                df = pd.read_csv(file_path)
                
                # Aggiungi la colonna context_type usando il mapping
                df['context_type'] = df['unique_id'].map(context_mapping)
                
                # Mappa context_type a valori binari: neutral=0, non_neutral=1
                context_type_to_binary = {
                    'neutral': 0,
                    'non_neutral': 1
                }
                df['context_type'] = df['context_type'].map(context_type_to_binary)
                
                # Rimuovi righe senza context_type valido
                df_filtered = df.dropna(subset=['context_type'])
                df_filtered['context_type'] = df_filtered['context_type'].astype(int)
                
                if len(df_filtered) == 0:
                    print(f"⚠ Warning: No valid context_type found for {dataset_name} - {model_name} - {prompting_method} - {input_type}")
                    continue
                
                # Valuta
                metrics = compute_metrics(df_filtered, accuracy_metric, precision_metric, recall_metric, f1_metric)
                
                # Salva i risultati
                results_data[dataset_name][model_name].append({
                    'file_path': file_path,
                    'input_type': input_type,
                    'technique': prompting_method,
                    'metrics': metrics
                })
                
                print(f"✓ Evaluated: {dataset_name} - {model_name} - {prompting_method} - {input_type}")
    
    # Stampa risultati in console
    print_console_results(results_data)
    
    print(f"\n" + "=" * 100)
    print(f"✓ Report generated successfully!")
    print("=" * 100)


if __name__ == "__main__":
    main()
