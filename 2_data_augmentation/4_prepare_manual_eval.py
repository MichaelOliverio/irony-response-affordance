import json
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Set

# Mapping degli ID per ogni configurazione
MAPPING = {
    "meta-llama_Llama-3.1-8B-Instruct_n_turn_1": '288f635e',
    "meta-llama_Llama-3.1-8B-Instruct_n_turn_2": '502e4213',
    "mistralai_Ministral-8B-Instruct-2410_n_turn_1": 'b3b3dcb3',
    "mistralai_Ministral-8B-Instruct-2410_n_turn_2": '087b3df3',
}

def format_dialogue(turns, dataset_name):
    """
    Formatta i turni del dialogo come JSON ben indentato.
    
    Args:
        turns: Lista di dict con le conversazioni
        dataset_name: Nome del dataset (CSC o MultiPICo)
    
    Returns:
        Stringa JSON formattata del dialogo
    """
    return json.dumps(turns, indent=2, ensure_ascii=False)

def find_common_unique_ids(base_dir: Path, dataset_name: str, label_column: str) -> Dict[int, Set[str]]:
    """
    Trova gli unique_id comuni a tutte le configurazioni per ogni classe di label.
    
    Args:
        base_dir: Directory dei file
        dataset_name: Nome del dataset (CSC o MultiPICo)
        label_column: Nome della colonna label
        
    Returns:
        Dict con {label: set of common unique_ids}
    """
    # Carica tutti i file e raccogli gli unique_id per label
    all_ids_by_label = {}  # {label: {model_key: set(unique_ids)}}
    
    for model_key in MAPPING.keys():
        filename = f"{dataset_name}_{model_key}_filtered.json"
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"⚠ File non trovato per analisi comuni: {filepath}")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filtra per turns validi
        df = pd.DataFrame([
            row for row in data
            if isinstance(row.get("dialogue"), dict)
            and "turns" in row["dialogue"]
            and row["dialogue"]["turns"]
        ])
        
        if df.empty:
            continue
        
        # Estrai label
        if label_column == "aggregate_irony_label":
            df[label_column] = df['annotations'].apply(
                lambda x: x.get('aggregate_irony_label') if isinstance(x, dict) else None
            )
        else:
            df[label_column] = df['annotations'].apply(
                lambda x: x.get('aggregate_sarcasm_label') if isinstance(x, dict) else None
            )
        
        df = df.dropna(subset=[label_column])
        
        # Raggruppa per label
        for label in [0, 1]:
            label_df = df[df[label_column] == label]
            unique_ids = set(label_df['unique_id'].tolist())
            
            if label not in all_ids_by_label:
                all_ids_by_label[label] = {}
            all_ids_by_label[label][model_key] = unique_ids
    
    # Trova gli ID comuni per ogni label
    common_ids_by_label = {}
    for label, model_dict in all_ids_by_label.items():
        if len(model_dict) == len(MAPPING):  # Tutti i modelli presenti
            # Intersezione di tutti i set
            common_ids = set.intersection(*model_dict.values())
            common_ids_by_label[label] = common_ids
            print(f"  Label {label}: {len(common_ids)} unique_id comuni")
        else:
            print(f"  ⚠ Label {label}: non tutti i modelli disponibili")
    
    return common_ids_by_label

def sample_common_dialogues(common_ids_by_label: Dict[int, Set[str]], n_per_class=10) -> Set[str]:
    """
    Campiona n unique_id per classe dagli ID comuni.
    
    Args:
        common_ids_by_label: Dict con {label: set of common unique_ids}
        n_per_class: Numero di campioni per classe
        
    Returns:
        Set di unique_id selezionati
    """
    selected_ids = set()
    
    for label, ids in common_ids_by_label.items():
        ids_list = list(ids)
        # Usa random_state per riproducibilità
        import random
        random.seed(42)
        random.shuffle(ids_list)
        
        # Prendi i primi n
        n_to_select = min(n_per_class, len(ids_list))
        selected_ids.update(ids_list[:n_to_select])
        print(f"  Selezionati {n_to_select} unique_id per label {label}")
    
    return selected_ids

def load_and_filter_data(base_dir, dataset_name, model_key, selected_ids: Set[str]):
    """
    Carica e filtra i dati da un file JSON usando gli unique_id selezionati.
    
    Args:
        base_dir: Directory base dove si trovano i file
        dataset_name: Nome del dataset (CSC o MultiPICo)
        model_key: Chiave del modello nel MAPPING
        selected_ids: Set di unique_id da includere
    
    Returns:
        DataFrame filtrato
    """
    # Costruisci il nome del file (con suffisso _filtered)
    filename = f"{dataset_name}_{model_key}_filtered.json"
    filepath = base_dir / filename
    
    if not filepath.exists():
        print(f"⚠ File non trovato: {filepath}")
        return None
    
    print(f"Caricamento: {filepath}")
    
    # Carica il file
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filtra righe con dialogue.turns e unique_id selezionati
    filtered_rows = [
        row for row in data
        if isinstance(row.get("dialogue"), dict)
        and "turns" in row["dialogue"]
        and row["dialogue"]["turns"]
        and row.get("unique_id") in selected_ids
    ]
    
    print(f"  Righe filtrate per unique_id selezionati: {len(filtered_rows)}")
    
    if not filtered_rows:
        return None
    
    df = pd.DataFrame(filtered_rows)
    
    # Determina la colonna label appropriata
    if dataset_name == "MultiPICo":
        label_column = "aggregate_irony_label"
    elif dataset_name == "CSC":
        label_column = "aggregate_sarcasm_label"
    else:
        print(f"⚠ Dataset sconosciuto: {dataset_name}")
        return None
    
    # Estrai la label dalle annotations
    if label_column == "aggregate_irony_label":
        df[label_column] = df['annotations'].apply(
            lambda x: x.get('aggregate_irony_label') if isinstance(x, dict) else None
        )
    else:  # aggregate_sarcasm_label
        df[label_column] = df['annotations'].apply(
            lambda x: x.get('aggregate_sarcasm_label') if isinstance(x, dict) else None
        )
    
    # Formatta il dialogo
    df['dialogue_formatted'] = df.apply(
        lambda row: format_dialogue(row['dialogue']['turns'], dataset_name), 
        axis=1
    )
    
    # Seleziona le colonne da mantenere
    if dataset_name == "MultiPICo":
        columns = ["unique_id", "dataset_name", "post", "reply", label_column, "dialogue_formatted"]
    else:  # CSC
        columns = ["unique_id", "dataset_name", "context", "reply", label_column, "dialogue_formatted"]
    
    df_result = df[columns].copy()
    
    # Rinomina la colonna dialogue_formatted
    df_result = df_result.rename(columns={"dialogue_formatted": "dialogue"})
    
    # Ordina per label e unique_id per consistenza
    df_result = df_result.sort_values(by=[label_column, "unique_id"])
    
    return df_result

def main():
    # Directory dei file filtrati
    base_dir = Path("human_annotations")
    csc_dir = base_dir / "csc_dialogues"
    multipico_dir = base_dir / "multipico_dialogues"

    # Directory di output
    output_base = Path("human_annotations/manual_eval_files")
    output_base.mkdir(exist_ok=True, parents=True)
    
    # File Excel unico con 8 sheet
    output_excel = output_base / "manual_evaluation_dialogues.xlsx"
    
    print("🚀 Inizio preparazione file per valutazione manuale\n")
    
    # FASE 1: Trova gli unique_id comuni per ogni dataset
    print("\n" + "="*70)
    print("FASE 1: Analisi unique_id comuni")
    print("="*70)
    
    print("\n📊 CSC Dataset - Ricerca unique_id comuni:")
    csc_common_ids = find_common_unique_ids(csc_dir, "CSC", "aggregate_sarcasm_label")
    csc_selected_ids = sample_common_dialogues(csc_common_ids, n_per_class=10)
    print(f"  Totale unique_id selezionati per CSC: {len(csc_selected_ids)}")
    
    print("\n📊 MultiPICo Dataset - Ricerca unique_id comuni:")
    multipico_common_ids = find_common_unique_ids(multipico_dir, "MultiPICo", "aggregate_irony_label")
    multipico_selected_ids = sample_common_dialogues(multipico_common_ids, n_per_class=10)
    print(f"  Totale unique_id selezionati per MultiPICo: {len(multipico_selected_ids)}")
    
    # FASE 2: Carica i dati con gli unique_id selezionati
    print("\n" + "="*70)
    print("FASE 2: Caricamento dati con unique_id selezionati")
    print("="*70)
    
    # Dizionario per contenere tutti i DataFrame
    all_sheets = {}
    
    # Processa ogni configurazione
    for model_key, sheet_id in MAPPING.items():
        print(f"\n{'='*70}")
        print(f"Processando: {model_key}")
        print(f"Sheet ID: {sheet_id}")
        print(f"{'='*70}")
        
        # CSC
        print("\n📂 CSC Dataset:")
        df_csc = load_and_filter_data(csc_dir, "CSC", model_key, csc_selected_ids)
        if df_csc is not None:
            sheet_name_csc = f"CSC_{sheet_id}"
            all_sheets[sheet_name_csc] = df_csc
            print(f"✓ Sheet '{sheet_name_csc}' pronto con {len(df_csc)} dialoghi")
        
        # MultiPICo
        print("\n📂 MultiPICo Dataset:")
        df_multipico = load_and_filter_data(multipico_dir, "MultiPICo", model_key, multipico_selected_ids)
        if df_multipico is not None:
            sheet_name_multipico = f"MultiPICo_{sheet_id}"
            all_sheets[sheet_name_multipico] = df_multipico
            print(f"✓ Sheet '{sheet_name_multipico}' pronto con {len(df_multipico)} dialoghi")
    
    # FASE 3: Salva il file Excel
    if all_sheets:
        print(f"\n{'='*70}")
        print(f"FASE 3: Salvataggio file Excel")
        print(f"{'='*70}")
        print(f"💾 File: {output_excel}\n")
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            for sheet_name, df in all_sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ Sheet '{sheet_name}' salvato ({len(df)} righe)")
        
        print(f"\n🎉 File Excel creato con successo!")
        print(f"📊 Totale sheet: {len(all_sheets)}")
        print(f"📁 Percorso: {output_excel.absolute()}")
    else:
        print("\n⚠ Nessun dato da salvare!")
        
if __name__ == "__main__":
    main()