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
        "LLama-3.1": "./output/cot_meta-llama_Llama-3.1-8B-Instruct_MultiPICo_n_turn_1_by_mistralai_Ministral-8B-Instruct-2410.json",
        "Ministral-8b": "./output/cot_mistralai_Ministral-8B-Instruct-2410_MultiPICo_n_turn_1_by_mistralai_Ministral-8B-Instruct-2410.json",
    }

    print("=" * 80)
    print("EVALUATION RESULTS - MultiPICo Dataset")
    print("=" * 80)

    for model_name, file_path in files.items():
        print(f"\n{model_name}")
        print("-" * 80)
        
        # Carica le predizioni dal JSON
        df = pd.read_json(file_path)
        
        # Converti model_answer in interi
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