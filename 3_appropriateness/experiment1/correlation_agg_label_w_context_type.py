import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import krippendorff
from pathlib import Path


def load_dataset(file_path):
    """Carica un dataset JSON."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_data_for_correlation(dataset):
    """Estrae context_type e aggregate_sarcasm_label per l'analisi di correlazione."""
    records = []
    
    entries = dataset.get('entries', dataset) if isinstance(dataset, dict) else dataset
    
    for entry in entries:
        context_type = entry.get('context_type')
        annotations = entry.get('annotations', {})
        aggregate_sarcasm_label = annotations.get('aggregate_sarcasm_label')
        unique_id = entry.get('unique_id')
        
        # Mappa context_type a valori binari
        if context_type == 'neutral':
            context_type_binary = 0
        elif context_type == 'non_neutral':
            context_type_binary = 1
        else:
            context_type_binary = None
        
        # Aggiungi solo se entrambi i valori sono validi
        if context_type_binary is not None and aggregate_sarcasm_label is not None:
            records.append({
                'unique_id': unique_id,
                'context_type': context_type_binary,
                'aggregate_sarcasm_label': aggregate_sarcasm_label
            })
    
    return pd.DataFrame(records)


def compute_correlations(df, dataset_name):
    """Calcola le correlazioni tra context_type e aggregate_sarcasm_label."""
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}".center(80))
    print(f"{'='*80}")
    
    # Statistiche descrittive
    print(f"\nTotal samples: {len(df)}")
    print(f"\nContext Type distribution:")
    print(f"  Neutral (0): {(df['context_type'] == 0).sum()} ({(df['context_type'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"  Non-neutral (1): {(df['context_type'] == 1).sum()} ({(df['context_type'] == 1).sum() / len(df) * 100:.2f}%)")
    
    print(f"\nAggregate Sarcasm Label distribution:")
    print(f"  Non-sarcastic (0): {(df['aggregate_sarcasm_label'] == 0).sum()} ({(df['aggregate_sarcasm_label'] == 0).sum() / len(df) * 100:.2f}%)")
    print(f"  Sarcastic (1): {(df['aggregate_sarcasm_label'] == 1).sum()} ({(df['aggregate_sarcasm_label'] == 1).sum() / len(df) * 100:.2f}%)")
    
    # Tabella di contingenza
    print(f"\nContingency Table:")
    contingency = pd.crosstab(
        df['context_type'], 
        df['aggregate_sarcasm_label'],
        rownames=['Context Type'],
        colnames=['Aggregate Sarcasm Label'],
        margins=True
    )
    print(contingency)
    
    # Calcola correlazioni
    print(f"\n{'-'*80}")
    print("INTER-RATER AGREEMENT ANALYSIS")
    print(f"{'-'*80}")
    
    # Cohen's Kappa
    cohen_kappa = cohen_kappa_score(df['context_type'], df['aggregate_sarcasm_label'])
    print(f"\nCohen's Kappa:")
    print(f"  Coefficient: {cohen_kappa:.4f}")
    
    # Interpretazione Cohen's Kappa
    if cohen_kappa < 0:
        interpretation = "Poor agreement (worse than random)"
    elif cohen_kappa < 0.20:
        interpretation = "Slight agreement"
    elif cohen_kappa < 0.40:
        interpretation = "Fair agreement"
    elif cohen_kappa < 0.60:
        interpretation = "Moderate agreement"
    elif cohen_kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    print(f"  Interpretation: {interpretation}")
    
    # Krippendorff's Alpha
    # Prepara i dati nel formato richiesto: matrice con righe = annotatori, colonne = item
    reliability_data = np.array([
        df['context_type'].values,
        df['aggregate_sarcasm_label'].values
    ])
    
    krippendorff_alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    print(f"\nKrippendorff's Alpha:")
    print(f"  Coefficient: {krippendorff_alpha:.4f}")
    
    # Interpretazione Krippendorff's Alpha
    if krippendorff_alpha < 0:
        interpretation_k = "Poor agreement (worse than random)"
    elif krippendorff_alpha < 0.20:
        interpretation_k = "Slight agreement"
    elif krippendorff_alpha < 0.40:
        interpretation_k = "Fair agreement"
    elif krippendorff_alpha < 0.60:
        interpretation_k = "Moderate agreement"
    elif krippendorff_alpha < 0.80:
        interpretation_k = "Substantial agreement"
    else:
        interpretation_k = "Almost perfect agreement"
    print(f"  Interpretation: {interpretation_k}")
    
    # Agreement percentage
    agreement = (df['context_type'] == df['aggregate_sarcasm_label']).sum()
    agreement_pct = agreement / len(df) * 100
    print(f"\nAgreement:")
    print(f"  Matching labels: {agreement} / {len(df)} ({agreement_pct:.2f}%)")
    
    return {
        'dataset': dataset_name,
        'n_samples': len(df),
        'cohen_kappa': cohen_kappa,
        'krippendorff_alpha': krippendorff_alpha,
        'agreement': agreement,
        'agreement_pct': agreement_pct
    }


def main():
    # Percorsi dei dataset
    datasets = {
        'CSC': '../datasets/CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2_filtered.json',
    }
    
    results = []
    
    for dataset_name, file_path in datasets.items():
        # Carica dataset
        dataset = load_dataset(file_path)
        
        # Estrai dati per correlazione
        df = extract_data_for_correlation(dataset)
        
        # Calcola correlazioni
        result = compute_correlations(df, dataset_name)
        results.append(result)
    
    # Riepilogo finale
    print(f"\n\n{'='*80}")
    print("SUMMARY".center(80))
    print(f"{'='*80}\n")
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("Analysis completed successfully!".center(80))
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
