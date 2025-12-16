import pandas as pd
from evaluate import load
from datetime import datetime
from pathlib import Path


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
    
    accuracy = accuracy_metric.compute(
        predictions=df["model_answer"], 
        references=df["label"]
    )["accuracy"]
    
    precision = precision_metric.compute(
        predictions=df["model_answer"], 
        references=df["label"], 
        pos_label=1,
        average="binary"
    )["precision"]
    
    recall = recall_metric.compute(
        predictions=df["model_answer"], 
        references=df["label"], 
        pos_label=1,
        average="binary"
    )["recall"]
    
    f1 = f1_metric.compute(
        predictions=df["model_answer"], 
        references=df["label"], 
        pos_label=1,
        average="binary"
    )["f1"]
    
    # Statistiche aggiuntive
    total = len(df)
    correct = (df["model_answer"] == df["label"]).sum()
    label_0 = (df["label"] == 0).sum()
    label_1 = (df["label"] == 1).sum()
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
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
            ],
            "mistralai_Ministral-8B-Instruct-2410": [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
            ],
        },
        'MultiPICo': { 
            "FacebookAI_xlm-roberta-base": [
                "./output/encoders/FacebookAI_xlm-roberta-base/MultiPICo/context/predictions/fine_tuning_FacebookAI_xlm-roberta-base_MultiPICo_context.csv",
                "./output/encoders/FacebookAI_xlm-roberta-base/MultiPICo/conversation/predictions/fine_tuning_FacebookAI_xlm-roberta-base_MultiPICo_conversation.csv",
            ],
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
            ],
            "mistralai_Ministral-8B-Instruct-2410": [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
            ],
        }
    }

    # Struttura per memorizzare tutti i risultati
    results_data = {}
    
    print("\nEvaluating models...\n")

    for dataset_name, dataset_files in files.items():
        results_data[dataset_name] = {}
        
        for model_name, file_paths in dataset_files.items():
            results_data[dataset_name][model_name] = []
            
            for file_path in file_paths:
                # Parsing delle informazioni dal file
                input_type, prompting_method = parse_file_info(file_path, dataset_name)
                
                # Carica e valuta
                df = pd.read_csv(file_path)
                metrics = compute_metrics(df, accuracy_metric, precision_metric, recall_metric, f1_metric)
                
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