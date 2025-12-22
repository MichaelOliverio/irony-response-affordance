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


def generate_summary_csv(results_data, output_path="./output/summary_results.csv"):
    """Genera un CSV con i risultati in formato tabellare come richiesto."""
    
    # Mapping degli input types
    input_type_mapping = {
        'context_reply': 'context_reply', #'C0 + MU2',
        'last_u1_message': 'last_u1_message', #'MU1',
        'conversation': 'conversation' #'T0 + T1 + MU1'
    }
    
    # Mapping dei modelli con short names
    model_mapping = {
        'Qwen_Qwen2.5-7B-Instruct': 'Qwen-2.5-7B',
        'mistralai_Ministral-8B-Instruct-2410': 'Ministral-8B',
        'google_gemma-2-9b-it': 'Gemma-2-9b',
        'FacebookAI_xlm-roberta-large': 'XLM-RoBERTa-large',
        'EuroBERT_EuroBERT-210m': 'EuroBERT-210M'
    }
    
    # Dataset mapping
    dataset_mapping = {
        'CSC': 'CANTRIP CSC',
        'MultiPICo': 'CANTRIP MP'
    }
    
    # Struttura per raccogliere i dati
    rows = []
    
    for dataset_name, dataset_results in results_data.items():
        dataset_label = dataset_mapping.get(dataset_name, dataset_name)
        
        # Per ogni tipo di input
        for input_key, input_label in input_type_mapping.items():
            row = {
                'Dataset': dataset_label,
                'Input': input_label
            }
            
            # Per ogni modello
            for model_full_name, model_short_name in model_mapping.items():
                if model_full_name not in dataset_results:
                    continue
                
                model_results = dataset_results[model_full_name]
                
                # Trova i risultati per questo input type
                for result in model_results:
                    if result['input_type'] == input_key:
                        technique = result['technique']
                        f1_score = result['metrics']['f1']
                        
                        # Crea la colonna appropriata
                        if technique == 'Zero-Shot':
                            col_name = f"{model_short_name}_ZS"
                        elif technique == 'Chain-of-Thought':
                            col_name = f"{model_short_name}_CT"
                        elif technique == 'Fine-Tuning':
                            col_name = f"{model_short_name}_FT"
                        else:
                            continue
                        
                        row[col_name] = round(f1_score, 2)
            
            if len(row) > 2:  # Se ci sono risultati oltre Dataset e Input
                rows.append(row)
    
    # Crea DataFrame e riordina le colonne
    df = pd.DataFrame(rows)
    
    # Definisci l'ordine delle colonne
    column_order = ['Dataset', 'Input']
    for model in ['Qwen-2.5-7B', 'Ministral-8B', 'Gemma-2-9b', 'XLM-RoBERTa-large', 'EuroBERT-210M']:
        for technique in ['ZS', 'CT', 'FT']:
            col = f"{model}_{technique}"
            if col in df.columns:
                column_order.append(col)
    
    # Riordina le colonne
    df = df[column_order]
    
    # Ordina per Dataset e Input
    df = df.sort_values(['Dataset', 'Input'])
    
    # Salva il CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Summary CSV saved to: {output_path}")
    
    return df

def main():
    # Carica le metriche da Hugging Face Evaluate
    print("Loading evaluation metrics...")
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    # File da valutare
    files = {
        'MultiPICo': { 
            #"FacebookAI_xlm-roberta-base": [
            #    "./output/encoders/FacebookAI_xlm-roberta-base/MultiPICo/context/predictions/fine_tuning_FacebookAI_xlm-roberta-base_MultiPICo_last_u1_message.csv",
            #    "./output/encoders/FacebookAI_xlm-roberta-base/MultiPICo/conversation/predictions/fine_tuning_FacebookAI_xlm-roberta-base_MultiPICo_conversation.csv",
            #],
            "FacebookAI_xlm-roberta-large": [
                "./output/encoders/FacebookAI_xlm-roberta-large/MultiPICo/context_reply/predictions/fine_tuning_FacebookAI_xlm-roberta-large_MultiPICo_context_reply.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/MultiPICo/context/predictions/fine_tuning_FacebookAI_xlm-roberta-large_MultiPICo_last_u1_message.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/MultiPICo/conversation/predictions/fine_tuning_FacebookAI_xlm-roberta-large_MultiPICo_conversation.csv",
            ],
            "EuroBERT_EuroBERT-210m": [
                "./output/encoders/EuroBERT_EuroBERT-210m/MultiPICo/context_reply/predictions/fine_tuning_EuroBERT_EuroBERT-210m_MultiPICo_context_reply.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/MultiPICo/context/predictions/fine_tuning_EuroBERT_EuroBERT-210m_MultiPICo_last_u1_message.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/MultiPICo/conversation/predictions/fine_tuning_EuroBERT_EuroBERT-210m_MultiPICo_conversation.csv",
            ],
            "google_gemma-2-9b-it": [
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_MultiPICo_context_reply.csv",
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_MultiPICo_last_u1_message.csv",
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_MultiPICo_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_MultiPICo_context_reply.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_MultiPICo_last_u1_message.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_MultiPICo_conversation.csv",
            ],
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_last_u1_message.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_last_u1_message.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
                #"./output/decoders/Qwen_Qwen2.5-7B-Instruct/fine_tuned/fine_tuning_Qwen_Qwen2.5-7B-Instruct_MultiPICo_last_u1_message.csv",
                #"./output/decoders/Qwen_Qwen2.5-7B-Instruct/fine_tuned/fine_tuning_Qwen_Qwen2.5-7B-Instruct_MultiPICo_conversation.csv",
            ],
            "mistralai_Ministral-8B-Instruct-2410": [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_last_u1_message.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_last_u1_message.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_MultiPICo_conversation.csv",

            ],
        },
        'CSC': {
            "EuroBERT_EuroBERT-210m": [
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/context_reply/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_context_reply.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/last_u1_message/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_last_u1_message.csv",
                #"./output/encoders/EuroBERT_EuroBERT-210m/CSC/context/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_context.csv",
                "./output/encoders/EuroBERT_EuroBERT-210m/CSC/conversation/predictions/fine_tuning_EuroBERT_EuroBERT-210m_CSC_conversation.csv",
            ],
            "FacebookAI_xlm-roberta-large": [
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/context_reply/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_context_reply.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/last_u1_message/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_last_u1_message.csv",
                # "./output/encoders/FacebookAI_xlm-roberta-large/CSC/context/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_context.csv",
                "./output/encoders/FacebookAI_xlm-roberta-large/CSC/conversation/predictions/fine_tuning_FacebookAI_xlm-roberta-large_CSC_conversation.csv",
            ],
            "google_gemma-2-9b-it": [
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_CSC_context_reply.csv",
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_CSC_last_u1_message.csv",
                #"./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_CSC_context.csv",
                "./output/decoders/google_gemma-2-9b-it/zero_shot_google_gemma-2-9b-it_CSC_conversation.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_CSC_context_reply.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_CSC_last_u1_message.csv",
                #"./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_CSC_context.csv",
                "./output/decoders/google_gemma-2-9b-it/chain_of_thought_google_gemma-2-9b-it_CSC_conversation.csv",
            ],
            "Qwen_Qwen2.5-7B-Instruct": [
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_last_u1_message.csv",
                #"./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/zero_shot_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_context_reply.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_last_u1_message.csv",
                #"./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_context.csv",
                "./output/decoders/Qwen_Qwen2.5-7B-Instruct/chain_of_thought_Qwen_Qwen2.5-7B-Instruct_CSC_conversation.csv",
            ],
            "mistralai_Ministral-8B-Instruct-2410": [
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_last_u1_message.csv",
                #"./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/zero_shot_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_context_reply.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_last_u1_message.csv",
                #"./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_context.csv",
                "./output/decoders/mistralai_Ministral-8B-Instruct-2410/chain_of_thought_mistralai_Ministral-8B-Instruct-2410_CSC_conversation.csv",
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
    
    # Genera il CSV di riepilogo
    summary_df = generate_summary_csv(results_data, "./output/summary_results.csv")
    
    print(f"\n" + "=" * 100)
    print(f"✓ Report generated successfully!")
    print("=" * 100)


if __name__ == "__main__":
    main()