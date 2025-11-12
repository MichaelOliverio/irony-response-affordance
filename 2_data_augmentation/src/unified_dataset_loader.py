import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from .config import UNIFIED_CORPUS_PATH, print_config

class UnifiedDatasetLoader:
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
        file_path = UNIFIED_CORPUS_PATH
        
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
            
            print(f"✓ Caricato unified_corpus da: {file_path}")
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
