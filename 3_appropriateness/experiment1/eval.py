import pandas as pd
from evaluate import load

def main():
    # Carica le metriche da Hugging Face Evaluate
    accuracy_metric = load("accuracy")
    precision_metric = load("precision")
    recall_metric = load("recall")
    f1_metric = load("f1")

    # File da valutare
    files = {
        'CSC': {
            "Llama-3.2-1B-Instruct": [
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_CSC_context.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_CSC_context_reply.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_CSC_conversation.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_CSC_context.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_CSC_context_reply.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_CSC_conversation.csv",
            ],
        },
        'MultiPICo': { 
            "Llama-3.2-1B-Instruct": [
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_context.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/zero_shot_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_conversation.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_context.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_context_reply.csv",
                "./output/decoders/meta-llama_Llama-3.2-1B-Instruct/chain_of_thought_meta-llama_Llama-3.2-1B-Instruct_MultiPICo_conversation.csv",
            ],
        }
    }

    for dataset_name, files in files.items():
        print("=" * 80)
        print(f"EVALUATION RESULTS - {dataset_name} Dataset")
        print("=" * 80)

        for model_name, file_paths in files.items():
            print(f"\n{model_name}")
            print("-" * 80)

            for file_path in file_paths:
                print(f"\nEvaluating file: {file_path}")

                # get type that is from {dataset_name} and .csv
                input_type = file_path.split(f"_{dataset_name}_")[1].split(".csv")[0]
                print(f"Input Type: {input_type}")
                
                # get prompting method from filename
                if "zero_shot" in file_path:
                    prompting_method = "Zero-Shot"
                elif "chain_of_thought" in file_path:
                    prompting_method = "Chain-of-Thought"
                else:
                    prompting_method = "Unknown"
                print(f"Prompting Method: {prompting_method}")

                # Carica le predizioni
                df = pd.read_csv(file_path)

                df["model_answer"] = df["model_answer"].astype(int)
                
                # Calcola le metriche
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
                
                # Stampa i risultati
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                
                # Statistiche aggiuntive
                total = len(df)
                correct = (df["model_answer"] == df["label"]).sum()
                label_0 = (df["label"] == 0).sum()
                label_1 = (df["label"] == 1).sum()
                pred_0 = (df["model_answer"] == 0).sum()
                pred_1 = (df["model_answer"] == 1).sum()
                
                print(f"\nTotal samples: {total}")
                print(f"Correct predictions: {correct}")
                print(f"Label distribution: 0={label_0}, 1={label_1}")
                print(f"Prediction distribution: 0={pred_0}, 1={pred_1}")

        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()