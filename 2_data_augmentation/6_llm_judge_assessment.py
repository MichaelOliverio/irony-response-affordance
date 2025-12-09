import pandas as pd
import numpy as np
import json
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import warnings

HUMAN_ANNOTATIONS_DIR = "human_annotations/manual_eval_files/human"
JUDGE_ANNOTATIONS_DIR = "human_annotations/manual_eval_files/judge"
RESULTS_DIR = "human_annotations/manual_eval_files/results"
PREPROCESSED_DIR = f"{RESULTS_DIR}/1_preprocessed"
#RANKING_DIR = f"{RESULTS_DIR}/2_ranking"
CORRELATION_DIR = f"{RESULTS_DIR}/3_correlation"
STATISTICAL_TESTS_DIR = f"{RESULTS_DIR}/4_statistical_tests"
LATIN_SQUARE_ASSIGNMENTS_FILE = "human_annotations/manual_eval_files/latin_square_assignments.json"

ANNOTATOR_IDS = ["annotator_1", "annotator_2", "annotator_3", "annotator_4"] # ["annotator_1", "annotator_2", "annotator_3", "annotator_4"]
JUDGE_MODELS = ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"] #"google/gemma-3-12b-it"
NUM_RECORD = None # None: tutti i record
DATASET_NAMES = ["csc", "multipico"] #"csc", multipico 
DIMENSIONS = {
    "Coherence": ['Logical Flow', 'Relevance', 'Continuity 1', 'Continuity 2'],
    "Engagingness": ['Informative', 'Emotional Engagement', 'Interest Level'],
    "Groundedness": ['Relevance 1', 'Relevance 2', 'Consistency 1', 'Consistency 2'],
    "Naturalness": ['Avoid repetition', 'Context relevance', 'Clarity', 'Word choice and tone 1', 'Word choice and tone 2']
}

# Pulisce le valutazioni convertendo 'Yes' in 1, 'No' in 0 e altro (indecisione) in 0.5
def clean_scores(df):
    for col in df.columns[3:19]:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else (0 if x == 'No' else 0.5))
    return df


def preprocess_datasets():
    """Preprocessa sia le annotazioni umane che quelle dei judge"""
    # Crea le directory se non esistono
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    #os.makedirs(RANKING_DIR, exist_ok=True)
    os.makedirs(CORRELATION_DIR, exist_ok=True)
    os.makedirs(STATISTICAL_TESTS_DIR, exist_ok=True)
    
    lsa = None
    with open(LATIN_SQUARE_ASSIGNMENTS_FILE, "r") as f:
        lsa = json.load(f)

    if lsa is None:
        raise ValueError("Latin square assignments file could not be loaded.")
    
    for dataset_name in DATASET_NAMES:
        # Preprocessa annotazioni umane
        preprocess_human_annotations(dataset_name, lsa)
        
        # Preprocessa annotazioni dei judge
        preprocess_judge_annotations(dataset_name, lsa)


def preprocess_human_annotations(dataset_name, lsa):
    """Preprocessa le annotazioni umane"""
    for annotator_id in ANNOTATOR_IDS:
        df = pd.read_excel(f"{HUMAN_ANNOTATIONS_DIR}/{annotator_id}.xlsx", sheet_name=dataset_name.upper())

        # pulizia del df
        if NUM_RECORD is not None:
            df = df.head(NUM_RECORD + 2)
        df = df.iloc[2:] # rimuovi le prime due righe di header
        df = df.reset_index(drop=True)
        df = clean_scores(df)

        # aggiunta dell'informazione del setting per ogni istanza
        unique_id_to_setting = lsa[dataset_name][annotator_id]['unique_id_to_setting']
        df['setting'] = df.iloc[:, 0].map(unique_id_to_setting) # Aggiungi la colonna 'setting' ai dataframe

        # rename delle colonne per uniformità
        df = df.rename(columns={
            'Unnamed: 0': 'unique_id',
            'Unnamed: 1': 'generated_turns',
            'Unnamed: 2': 'original_turn',
            'Coherence': 'Logical Flow',
            'Unnamed: 4': 'Relevance',
            'Unnamed: 5': 'Continuity 1',
            'Unnamed: 6': 'Continuity 2',
            'Engagingness': 'Informative',
            'Unnamed: 8': 'Emotional Engagement',
            'Unnamed: 9': 'Interest Level',
            'Groundedness': 'Relevance 1',
            'Unnamed: 11': 'Relevance 2',
            'Unnamed: 12': 'Consistency 1',
            'Unnamed: 13': 'Consistency 2',
            'Naturalness': 'Avoid repetition',
            'Unnamed: 15': 'Context relevance',
            'Unnamed: 16': 'Clarity',
            'Unnamed: 17': 'Word choice and tone 1',
            'Unnamed: 18': 'Word choice and tone 2',
            'Unnamed: 19': 'notes'
        })

        # salva in un file json
        output_data = df.to_dict(orient='records')
        output_file = f"{PREPROCESSED_DIR}/{dataset_name}_{annotator_id}_preprocessed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"     Salvato: {dataset_name}_{annotator_id}_preprocessed.json")


def preprocess_judge_annotations(dataset_name, lsa):
    """Preprocessa le annotazioni dei judge convertendole nel formato delle annotazioni umane"""
    judge_file = f"{JUDGE_ANNOTATIONS_DIR}/prompt6_judge_scores_{dataset_name}_all_instances_for_llm_judge.json" # il 3 sembra il migliore
    
    with open(judge_file, "r", encoding="utf-8") as f:
        judge_data = json.load(f)
    
    # Ottieni gli unique_id delle istanze annotate dagli annotatori umani
    human_unique_ids = set()
    for annotator_id in ANNOTATOR_IDS:
        annotator_data = lsa[dataset_name][annotator_id]
        unique_ids = list(annotator_data['unique_id_to_setting'].keys())
        if NUM_RECORD is not None:
            unique_ids = unique_ids[:NUM_RECORD]
        human_unique_ids.update(unique_ids)
    
    print(f"     Istanze annotate dagli annotatori umani: {len(human_unique_ids)}")
    
    for judge_model in JUDGE_MODELS:
        if judge_model not in judge_data["judge"]:
            print(f"     Judge model {judge_model} not found in {judge_file}")
            continue
        
        annotations = judge_data["judge"][judge_model]
        
        # Crea un DataFrame con la stessa struttura delle annotazioni umane
        # Filtra solo le istanze annotate dagli annotatori umani
        rows = []
        for annotation in annotations:
            unique_id = annotation['unique_id']
            
            # Prendi solo le istanze che sono state annotate dagli annotatori umani
            if unique_id not in human_unique_ids:
                continue
                
            row = {
                'unique_id': unique_id,
                'dataset_name': annotation['dataset_name'],
                'setting': annotation['generation_setting']
            }
            
            # Aggiungi gli score per ogni dimensione
            for dimension, questions in DIMENSIONS.items():
                if dimension in annotation:
                    scores = annotation[dimension]['scores']
                    for question in questions:
                        row[question] = scores.get(question, 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Salva in un file json
        judge_name = judge_model.replace('/', '_')
        output_data = df.to_dict(orient='records')
        output_file = f"{PREPROCESSED_DIR}/{dataset_name}_judge_{judge_name}_preprocessed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"     Salvato: {dataset_name}_judge_{judge_name}_preprocessed.json ({len(df)} istanze)")


def correlation(dataset_name):
    """Calcola la correlazione tra annotatori umani vs judge"""
    
    dimensions = ['Coherence', 'Engagingness', 'Groundedness', 'Naturalness', 'Total']
    
    # Carica tutti i dati degli annotatori umani
    human_dfs = {}
    for annotator_id in ANNOTATOR_IDS:
        human_file = f"{PREPROCESSED_DIR}/{dataset_name}_{annotator_id}_preprocessed.json"
        with open(human_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Aggrega i punteggi per dimensione (media)
        df['Coherence_total'] = df[DIMENSIONS['Coherence']].mean(axis=1)
        df['Engagingness_total'] = df[DIMENSIONS['Engagingness']].mean(axis=1)
        df['Groundedness_total'] = df[DIMENSIONS['Groundedness']].mean(axis=1)
        df['Naturalness_total'] = df[DIMENSIONS['Naturalness']].mean(axis=1)
        df['Total_total'] = df[['Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total']].mean(axis=1)
        
        # Seleziona solo le colonne necessarie
        df = df[['unique_id', 'setting', 'Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total', 'Total_total']]
        human_dfs[annotator_id] = df
    
    # ===== CORRELAZIONE ANNOTATORI UMANI vs JUDGE =====
    print("\n" + "="*70)
    print("     CORRELAZIONE ANNOTATORI UMANI vs JUDGE")
    print("="*70)
    
    # Per ogni coppia annotatore umano - judge
    for annotator_id in ANNOTATOR_IDS:
        # Usa il dataframe già caricato
        df_human = human_dfs[annotator_id]
        
        for judge_model in JUDGE_MODELS:
            judge_name = judge_model.replace('/', '_')
            judge_file = f"{PREPROCESSED_DIR}/{dataset_name}_judge_{judge_name}_preprocessed.json"
            
            try:
                with open(judge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df_judge = pd.DataFrame(data)
            except FileNotFoundError:
                print(f"     File non trovato: {judge_file}")
                continue
            
            # Aggrega i punteggi per dimensione (media)
            df_judge['Coherence_total'] = df_judge[DIMENSIONS['Coherence']].mean(axis=1)
            df_judge['Engagingness_total'] = df_judge[DIMENSIONS['Engagingness']].mean(axis=1)
            df_judge['Groundedness_total'] = df_judge[DIMENSIONS['Groundedness']].mean(axis=1)
            df_judge['Naturalness_total'] = df_judge[DIMENSIONS['Naturalness']].mean(axis=1)
            df_judge['Total_total'] = df_judge[['Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total']].mean(axis=1)
            
            # Seleziona solo le colonne necessarie
            df_judge = df_judge[['unique_id', 'setting', 'Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total', 'Total_total']]
            
            # Merge dei dataframe
            df_merge = pd.merge(df_human, df_judge, on=['unique_id', 'setting'], suffixes=('_human', '_judge'))
            
            print(f"\n     Correlazione: {annotator_id} vs {judge_model}")
            print(f"   Istanze comuni: {len(df_merge)}")
            
            # Calcola correlazione per setting
            results = []
            for setting, group in df_merge.groupby('setting'):
                row = {'setting': setting}
                for dim in dimensions:
                    col_human = f"{dim}_total_human"
                    col_judge = f"{dim}_total_judge"
                    
                    # Controlla se ci sono abbastanza punti e varianza
                    if len(group) > 1 and group[col_human].std() > 0 and group[col_judge].std() > 0:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            r, p = spearmanr(group[col_human], group[col_judge])
                        row[f'{dim.lower()}_r'] = r
                        row[f'{dim.lower()}_p'] = p
                    else:
                        row[f'{dim.lower()}_r'] = np.nan
                        row[f'{dim.lower()}_p'] = np.nan
                results.append(row)
            
            df_spearman = pd.DataFrame(results)
            
            # Stampa risultati ordinati per setting
            print(f"\n   Correlazioni di Spearman per setting:")
            for _, row in df_spearman.sort_values('setting').iterrows():
                print(f"\n   {row['setting']}:")
                for dim in dimensions:
                    r = row[f'{dim.lower()}_r']
                    p = row[f'{dim.lower()}_p']
                    if not np.isnan(r):
                        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"      {dim:15s}: r={r:6.3f}, p={p:6.4f} {significance}")
                    else:
                        print(f"      {dim:15s}: N/A")
            
            # Salva risultati in JSON nella cartella correlation
            output_file = f"{CORRELATION_DIR}/{dataset_name}_correlation_human_{annotator_id}_vs_judge_{judge_name}.json"
            results_dict = df_spearman.to_dict(orient='records')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2)
            print(f"\n        Salvato: {output_file}")
    
    # ===== CORRELAZIONE TOTALE (TUTTE LE ANNOTAZIONI AGGREGATE) =====
    print("\n" + "="*70)
    print("     CORRELAZIONE TOTALE (TUTTI GLI ANNOTATORI AGGREGATI)")
    print("="*70)
    
    for judge_model in JUDGE_MODELS:
        judge_name = judge_model.replace('/', '_')
        judge_file = f"{PREPROCESSED_DIR}/{dataset_name}_judge_{judge_name}_preprocessed.json"
        
        try:
            with open(judge_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_judge = pd.DataFrame(data)
        except FileNotFoundError:
            print(f"     File non trovato: {judge_file}")
            continue
        
        # Aggrega i punteggi per dimensione (media)
        df_judge['Coherence_total'] = df_judge[DIMENSIONS['Coherence']].mean(axis=1)
        df_judge['Engagingness_total'] = df_judge[DIMENSIONS['Engagingness']].mean(axis=1)
        df_judge['Groundedness_total'] = df_judge[DIMENSIONS['Groundedness']].mean(axis=1)
        df_judge['Naturalness_total'] = df_judge[DIMENSIONS['Naturalness']].mean(axis=1)
        df_judge['Total_total'] = df_judge[['Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total']].mean(axis=1)
        
        # Seleziona solo le colonne necessarie
        df_judge = df_judge[['unique_id', 'setting', 'Coherence_total', 'Engagingness_total', 'Groundedness_total', 'Naturalness_total', 'Total_total']]
        
        # Aggrega tutte le annotazioni umane
        all_human_data = []
        for annotator_id in ANNOTATOR_IDS:
            df_human = human_dfs[annotator_id].copy()
            df_human['annotator'] = annotator_id
            all_human_data.append(df_human)
        
        df_all_human = pd.concat(all_human_data, ignore_index=True)
        
        # Merge con i dati del judge (ogni istanza del judge viene matchata con tutte le annotazioni umane)
        df_merge = pd.merge(df_all_human, df_judge, on=['unique_id', 'setting'], suffixes=('_human', '_judge'))
        
        print(f"\n     Correlazione TOTALE: Tutti gli annotatori vs {judge_model}")
        print(f"   Istanze totali: {len(df_merge)} (annotazioni umane aggregate)")
        
        # Calcola correlazione complessiva per tutte le istanze
        print(f"\n   Correlazioni di Spearman COMPLESSIVE:")
        overall_results = {}
        for dim in dimensions:
            col_human = f"{dim}_total_human"
            col_judge = f"{dim}_total_judge"
            
            # Controlla se ci sono abbastanza punti e varianza
            if len(df_merge) > 1 and df_merge[col_human].std() > 0 and df_merge[col_judge].std() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r, p = spearmanr(df_merge[col_human], df_merge[col_judge])
                overall_results[dim] = {'r': r, 'p': p, 'n': len(df_merge)}
                significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"      {dim:15s}: r={r:6.3f}, p={p:6.4f} {significance} (n={len(df_merge)})")
            else:
                overall_results[dim] = {'r': np.nan, 'p': np.nan, 'n': len(df_merge)}
                print(f"      {dim:15s}: N/A")
        
        # Calcola correlazione per setting (con tutte le annotazioni aggregate)
        results_by_setting = []
        for setting, group in df_merge.groupby('setting'):
            row = {'setting': setting, 'n': len(group)}
            for dim in dimensions:
                col_human = f"{dim}_total_human"
                col_judge = f"{dim}_total_judge"
                
                # Controlla se ci sono abbastanza punti e varianza
                if len(group) > 1 and group[col_human].std() > 0 and group[col_judge].std() > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r, p = spearmanr(group[col_human], group[col_judge])
                    row[f'{dim.lower()}_r'] = r
                    row[f'{dim.lower()}_p'] = p
                else:
                    row[f'{dim.lower()}_r'] = np.nan
                    row[f'{dim.lower()}_p'] = np.nan
            results_by_setting.append(row)
        
        df_spearman_by_setting = pd.DataFrame(results_by_setting)
        
        #print(f"\n   Correlazioni di Spearman per setting (tutte le annotazioni aggregate):")
        #for _, row in df_spearman_by_setting.sort_values('setting').iterrows():
        #    print(f"\n   {row['setting']} (n={row['n']}):")
        #    for dim in dimensions:
        #        r = row[f'{dim.lower()}_r']
        #        p = row[f'{dim.lower()}_p']
        #        if not np.isnan(r):
        #            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        #            print(f"      {dim:15s}: r={r:6.3f}, p={p:6.4f} {significance}")
        #        else:
        #            print(f"      {dim:15s}: N/A")
        
        # Salva risultati aggregati in JSON
        output_overall = f"{CORRELATION_DIR}/{dataset_name}_correlation_TOTAL_human_vs_judge_{judge_name}.json"
        output_data = {
            'overall': overall_results,
            'by_setting': df_spearman_by_setting.to_dict(orient='records')
        }
        with open(output_overall, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n        Salvato: {output_overall}")


def statistical_tests(dataset_name):
    """
    Esegue test statistici (ANOVA + Tukey HSD) per determinare quale setting è significativamente migliore.
    Utilizza un Latin Square Design con gli annotatori come blocchi.
    """
    print("\n" + "="*70)
    print("     TEST STATISTICI - ANOVA (Latin Square Design)")
    print("="*70)
    
    # Carica tutti i dati degli annotatori umani
    all_data = []
    for annotator_id in ANNOTATOR_IDS:
        human_file = f"{PREPROCESSED_DIR}/{dataset_name}_{annotator_id}_preprocessed.json"
        with open(human_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Aggrega i punteggi per dimensione (media)
        df['Coherence_avg'] = df[DIMENSIONS['Coherence']].mean(axis=1)
        df['Engagingness_avg'] = df[DIMENSIONS['Engagingness']].mean(axis=1)
        df['Groundedness_avg'] = df[DIMENSIONS['Groundedness']].mean(axis=1)
        df['Naturalness_avg'] = df[DIMENSIONS['Naturalness']].mean(axis=1)
        
        # Media pesata (0.3, 0.1, 0.3, 0.3)
        df['Weighted_avg'] = (
            df['Coherence_avg'] * 0.3 +
            df['Engagingness_avg'] * 0.1 +
            df['Groundedness_avg'] * 0.3 +
            df['Naturalness_avg'] * 0.3
        )
        
        df['annotator'] = annotator_id
        all_data.append(df[['unique_id', 'setting', 'annotator', 'Coherence_avg', 'Engagingness_avg', 
                            'Groundedness_avg', 'Naturalness_avg', 'Weighted_avg']])
    
    # Concatena tutti i dati
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n     Dati caricati: {len(df_combined)} osservazioni totali")
    print(f"   Annotatori: {df_combined['annotator'].nunique()}")
    print(f"   Settings: {df_combined['setting'].nunique()}")
    
    # ===== ANOVA PER OGNI DIMENSIONE =====
    dimensions_to_test = ['Coherence_avg', 'Engagingness_avg', 'Groundedness_avg', 'Naturalness_avg', 'Weighted_avg']
    dimension_names = ['Coherence', 'Engagingness', 'Groundedness', 'Naturalness', 'Weighted Average']
    
    results_summary = []
    
    for dim_col, dim_name in zip(dimensions_to_test, dimension_names):
        print(f"\n{'='*70}")
        print(f"     {dim_name}")
        print('='*70)
        
        # Modello ANOVA con annotatore come blocco
        formula = f'{dim_col} ~ C(setting) + C(annotator)'
        model = ols(formula, data=df_combined).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print(f"\n     ANOVA Table:")
        print(anova_table)
        
        # Estrai p-value per setting
        p_value_setting = anova_table.loc['C(setting)', 'PR(>F)']
        f_value_setting = anova_table.loc['C(setting)', 'F']
        
        # Salva risultati (converti in tipi Python nativi per JSON)
        result = {
            'dimension': dim_name,
            'F_statistic': float(f_value_setting),
            'p_value': float(p_value_setting),
            'significant': bool(p_value_setting < 0.05)
        }
        
        # Calcola sempre il ranking dei settings (media più alta)
        setting_means = df_combined.groupby('setting')[dim_col].mean().sort_values(ascending=False)
        print(f"\n     Ranking dei settings (dal migliore):")
        for i, (setting, mean_score) in enumerate(setting_means.items(), 1):
            print(f"   {i}. {setting}: {mean_score:.4f}")
        
        # Converti setting_means in dict con valori float
        result['setting_means'] = {str(k): float(v) for k, v in setting_means.items()}
        result['best_setting'] = str(setting_means.index[0])
        
        # Se significativo, esegui Tukey HSD
        if p_value_setting < 0.05:
            print(f"\n     Differenza significativa trovata (p={p_value_setting:.4f})")
            print(f"\n     Post-hoc test (Tukey HSD):")
            
            tukey = pairwise_tukeyhsd(endog=df_combined[dim_col], 
                                     groups=df_combined['setting'], 
                                     alpha=0.05)
            print(tukey.summary())
            
            # Converti risultati Tukey in formato leggibile (con conversione a tipi Python nativi)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            # Converti tutti i valori in tipi Python nativi
            tukey_records = []
            for record in tukey_df.to_dict(orient='records'):
                converted_record = {}
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        converted_record[key] = float(value)
                    elif isinstance(value, np.bool_):
                        converted_record[key] = bool(value)
                    else:
                        converted_record[key] = value
                tukey_records.append(converted_record)
            result['tukey_results'] = tukey_records
        else:
            print(f"\n     Nessuna differenza significativa (p={p_value_setting:.4f})")
            result['tukey_results'] = None
        
        results_summary.append(result)
    
    # Salva risultati in JSON
    output_file = f"{STATISTICAL_TESTS_DIR}/{dataset_name}_anova_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n     Salvato: {output_file}")
    
    return results_summary


def main():
    print("\n" + "="*70)
    print("     PREPROCESSING DATASETS")
    print("="*70)
    preprocess_datasets()
    
    for dataset_name in DATASET_NAMES:
        print(f"\n" + "="*70)
        print(f"     ANALISI DELLE VALUTAZIONI PER IL DATASET: {dataset_name.upper()}")
        print("="*70)
        correlation(dataset_name)
        #statistical_tests(dataset_name)


if __name__ == "__main__":
    main()