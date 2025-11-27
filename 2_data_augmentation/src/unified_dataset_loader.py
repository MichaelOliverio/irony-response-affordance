import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from .config import UNIFIED_CORPUS_PATH
from sklearn.model_selection import train_test_split

class UnifiedDatasetLoader:
    @staticmethod
    def load_test_set(dataset_name='MultiPICo', language='en', limit=None):
        """
        Carica il dataset dal unified corpus, mantenendo gli unique_id anche negli split.
        """
        if dataset_name == 'MultiPICo':
            file_path = UNIFIED_CORPUS_PATH / "multipico_benchmark.json"
        elif dataset_name == 'CSC':
            file_path = UNIFIED_CORPUS_PATH / "csc_benchmark.json"
        else:
            raise ValueError(f"Dataset non supportato: {dataset_name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            unified_corpus = json.load(f)
        print(f"Loaded unified corpus with {len(unified_corpus)} entries")
        
        df = pd.DataFrame(unified_corpus)
        
        # Filtra per lingua e dataset
        df_filtered = df[(df['language'] == language) & (df['dataset_name'] == dataset_name)]
        print(f"Filtered {dataset_name} {language.upper()} entries: {len(df_filtered)}")
        
        # Limita il numero di entries se necessario
        if limit is not None:
            df_filtered = df_filtered.head(limit)
            print(f"Using {len(df_filtered)} entries (limited)")

        # Seleziona solo le colonne utili
        df_filtered['label'] = df_filtered['annotations'].apply(
            lambda x: x.get('irony_appropriateness_label', None)
        )
        df_filtered = df_filtered.drop(columns=['annotations'])

        print(f"Entries with label 1: {len(df_filtered[df_filtered['label'] == 1])}")
        print(f"Entries with label 0: {len(df_filtered[df_filtered['label'] == 0])}")

        # Includiamo anche gli ID nello split
        train_df, temp_df = train_test_split(
            df_filtered,
            test_size=0.3,
            random_state=42,
            stratify=df_filtered['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )

        # Mostra distribuzioni
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f'{name} len: {len(split_df)} | label 1: {len(split_df[split_df["label"]==1])} | label 0: {len(split_df[split_df["label"]==0])}')

        return test_df.reset_index(drop=True)

    @staticmethod
    def load_dataset(datasets: List[str], languages: List[str]) -> pd.DataFrame:
        """
        Carica il dataset unificato.
        
        Args:
            datasets: Lista dei nomi dei dataset da filtrare
            languages: Lista delle lingue da filtrare
            file_path: Path opzionale al file unified_corpus.json. Se None, usa UNIFIED_CORPUS_PATH dalla config.
        
        Returns:
            DataFrame filtrato
        """
        if 'MultiPICo' in datasets:
            file_path = UNIFIED_CORPUS_PATH / "multipico_benchmark.json"
        elif 'CSC' in datasets:
            file_path = UNIFIED_CORPUS_PATH / "csc_benchmark.json"
        else:
            raise ValueError(f"Dataset non supportato: {datasets}")
        
        # Verifica che il file esista
        if not file_path.exists():
            raise FileNotFoundError(
                f"File non trovato: {file_path}\n"
                f"Path assoluto: {file_path.absolute()}\n"
                f"Verifica che il file esista in questa posizione."
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                unified_corpus = json.load(f)
            df = pd.DataFrame(unified_corpus)
            
            print(f"Caricato unified_corpus da: {file_path}")
            print(f"  Totale righe: {len(df)}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Errore nel parsing JSON del file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Errore durante il caricamento del file {file_path}: {e}")

        # Filtra per dataset e lingue
        df_filtered = df[df['dataset_name'].isin(datasets) & df['language'].isin(languages)]
        
        print(f"  Righe dopo filtro (datasets={datasets}, languages={languages}): {len(df_filtered)}")
        # print per ogni dataset quanti esempi ci sono
        for dataset in datasets:
            count = len(df_filtered[df_filtered['dataset_name'] == dataset])
            print(f"    {dataset}: {count} righe")
        
        return df_filtered
