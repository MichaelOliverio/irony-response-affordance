import json
import pandas as pd
from pathlib import Path
from typing import Dict, Set, List
import random
from nltk.tokenize import word_tokenize

# Lista dei setting per il Latin Square
SETTINGS = [
    "meta-llama_Llama-3.1-8B-Instruct_n_turn_1",
    "meta-llama_Llama-3.1-8B-Instruct_n_turn_2",
    "mistralai_Ministral-8B-Instruct-2410_n_turn_1",
    "mistralai_Ministral-8B-Instruct-2410_n_turn_2",
]

NUM_ANNOTATORS = 4

def find_common_unique_ids(base_dir: Path, dataset_name: str, label_column: str, min_tokens: int = None) -> Dict[int, Set[str]]:
    """
    Trova gli unique_id comuni a tutte le configurazioni, separati per label.
    Per MultiPICo filtra anche per numero minimo di token.
    
    Args:
        base_dir: Directory dei file
        dataset_name: Nome del dataset (CSC o MultiPICo)
        label_column: Nome della colonna label
        min_tokens: Numero minimo di token per post+reply (solo MultiPICo)
        
    Returns:
        Dict {label: Set di unique_id comuni}
    """
    all_ids_by_model_and_label = {}  # {model_key: {label: set(unique_ids)}}
    
    for model_key in SETTINGS:
        filename = f"{dataset_name}_{model_key}_filtered.json"
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"⚠ File non trovato: {filepath}")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filtra per turns validi e raggruppa per label
        ids_by_label = {0: set(), 1: set()}
        
        for row in data:
            if not (isinstance(row.get("dialogue"), dict) 
                    and "turns" in row["dialogue"] 
                    and row["dialogue"]["turns"]):
                continue
            
            # Filtro per MultiPICo: >= 10 token (post + reply)
            if dataset_name == "MultiPICo" and min_tokens:
                post_tokens = len(word_tokenize(row.get('post', '')))
                reply_tokens = len(word_tokenize(row.get('reply', '')))
                if post_tokens + reply_tokens < min_tokens:
                    continue
            
            # Estrai label
            annotations = row.get('annotations', {})
            if isinstance(annotations, dict):
                label = annotations.get(label_column)
                if label in [0, 1]:
                    ids_by_label[label].add(row['unique_id'])
        
        all_ids_by_model_and_label[model_key] = ids_by_label
        print(f"  {model_key}: label 0={len(ids_by_label[0])}, label 1={len(ids_by_label[1])}")
    
    # Intersezione di tutti i set per ogni label
    common_ids_by_label = {}
    if len(all_ids_by_model_and_label) == len(SETTINGS):
        for label in [0, 1]:
            ids_sets = [model_data[label] for model_data in all_ids_by_model_and_label.values()]
            common_ids = set.intersection(*ids_sets)
            common_ids_by_label[label] = common_ids
            print(f"  ✓ Label {label}: {len(common_ids)} unique_id comuni")
        return common_ids_by_label
    else:
        print(f"  ⚠ Non tutti i modelli disponibili")
        return {}

def sample_instances(common_ids_by_label: Dict[int, Set[str]], n_per_label: int = 30) -> List[str]:
    """
    Campiona n_per_label unique_id per ogni label dagli ID comuni.
    
    Args:
        common_ids_by_label: Dict {label: Set di unique_id comuni}
        n_per_label: Numero di istanze per label (default 30)
        
    Returns:
        Lista di unique_id selezionati (bilanciata)
    """
    selected = []
    
    for label in [0, 1]:
        # Usa sorted() per garantire ordine deterministico
        ids_list = sorted(list(common_ids_by_label.get(label, set())))
        random.shuffle(ids_list)
        
        n_to_select = min(n_per_label, len(ids_list))
        selected.extend(ids_list[:n_to_select])
        print(f"  Label {label}: selezionate {n_to_select} istanze")
    
    print(f"  Totale: {len(selected)} istanze")
    return selected

def create_latin_square_assignment(instances: List[str], settings: List[str], n_annotators: int) -> Dict[int, Dict[str, str]]:
    """
    Crea l'assegnamento Latin Square: ogni annotatore vede tutte le istanze
    ma con setting diversi.
    
    Args:
        instances: Lista di unique_id (60 istanze)
        settings: Lista dei 4 setting
        n_annotators: Numero di annotatori (4)
        
    Returns:
        Dict {annotator_id: {unique_id: setting}}
    """
    assignment = {i: {} for i in range(n_annotators)}
    
    # Dividi le istanze in gruppi da 15 (uno per setting)
    instances_per_setting = len(instances) // len(settings)
    
    for i, instance_id in enumerate(instances):
        # Determina a quale gruppo appartiene questa istanza
        group = i // instances_per_setting
        
        # Per ogni annotatore, assegna un setting diverso usando il Latin Square
        for annotator in range(n_annotators):
            # Rotazione ciclica: annotatore 0 usa setting[group], 
            # annotatore 1 usa setting[(group+1)%4], ecc.
            setting_idx = (group + annotator) % len(settings)
            assigned_setting = settings[setting_idx]
            assignment[annotator][instance_id] = assigned_setting
    
    return assignment

def load_data_for_annotator(base_dir: Path, dataset_name: str, instances: List[str], 
                            assignment: Dict[str, str]) -> pd.DataFrame:
    """
    Carica i dati per un annotatore specifico secondo l'assegnamento Latin Square.
    
    Args:
        base_dir: Directory base
        dataset_name: Nome del dataset (CSC o MultiPICo)
        instances: Lista di unique_id da caricare
        assignment: Dict {unique_id: setting} per questo annotatore
        
    Returns:
        DataFrame con i dati combinati da tutti i setting necessari
    """
    all_rows = []
    
    # Raggruppa le istanze per setting
    instances_by_setting = {}
    for instance_id in instances:
        setting = assignment[instance_id]
        if setting not in instances_by_setting:
            instances_by_setting[setting] = []
        instances_by_setting[setting].append(instance_id)
    
    # Carica i dati per ogni setting
    for setting, instance_ids in instances_by_setting.items():
        model_key = setting
        filename = f"{dataset_name}_{model_key}_filtered.json"
        filepath = base_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ File non trovato: {filepath}")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filtra per unique_id di questo setting e aggiungi il setting
        for row in data:
            if row.get("unique_id") in instance_ids:
                if isinstance(row.get("dialogue"), dict) and "turns" in row["dialogue"]:
                    # Aggiungi informazione sul setting
                    row['generation_setting'] = setting
                    all_rows.append(row)
    
    if not all_rows:
        return None
    
    df = pd.DataFrame(all_rows)
    
    # Determina la colonna label e i campi per original_turn
    is_multipico = dataset_name == "MultiPICo"
    label_column = "aggregate_irony_label" if is_multipico else "aggregate_sarcasm_label"
    
    # Estrai label
    df[label_column] = df['annotations'].apply(
        lambda x: x.get(label_column) if isinstance(x, dict) else None
    )
    
    # Formatta dialogo
    df['dialogue'] = df['dialogue'].apply(
        lambda d: json.dumps(d['turns'], indent=2, ensure_ascii=False)
    )
    
    # Crea original_turn
    if is_multipico:
        df['original_turn'] = df.apply(
            lambda row: json.dumps({"post": row['post'], "reply": row['reply']}, 
                                 indent=2, ensure_ascii=False),
            axis=1
        )
    else:
        df['original_turn'] = df.apply(
            lambda row: json.dumps({"context": row['context'], "reply": row['reply']}, 
                                 indent=2, ensure_ascii=False),
            axis=1
        )
    
    # Seleziona e ordina
    columns = ["unique_id", "dataset_name", "generation_setting", "original_turn", label_column, "dialogue"]
    return df[columns].sort_values(by="unique_id")

def main():
    # Imposta il seed per riproducibilità
    random.seed(42)
    
    # Directory dei file filtrati
    base_dir = Path("human_annotations")
    csc_dir = base_dir / "csc_dialogues"
    multipico_dir = base_dir / "multipico_dialogues"

    # Directory di output
    output_base = Path("human_annotations/manual_eval_files")
    output_base.mkdir(exist_ok=True, parents=True)
    
    print("🚀 Inizio preparazione file per valutazione manuale (Latin Square Design)\n")
    print("   [Seed random: 42 - per riproducibilità]")
    
    # FASE 1: Trova gli unique_id comuni
    print("\n" + "="*70)
    print("FASE 1: Analisi unique_id comuni")
    print("="*70)
    
    print("\n📊 CSC Dataset:")
    csc_common_ids = find_common_unique_ids(csc_dir, "CSC", "aggregate_sarcasm_label")
    csc_instances = sample_instances(csc_common_ids, n_per_label=30)
    print(f"  ✓ {len(csc_instances)} istanze selezionate per CSC")
    
    print("\n📊 MultiPICo Dataset (filtro >= 10 token):")
    multipico_common_ids = find_common_unique_ids(multipico_dir, "MultiPICo", "aggregate_irony_label", min_tokens=10)
    multipico_instances = sample_instances(multipico_common_ids, n_per_label=30)
    print(f"  ✓ {len(multipico_instances)} istanze selezionate per MultiPICo")
    
    # FASE 2: Crea Latin Square assignment
    print("\n" + "="*70)
    print("FASE 2: Creazione Latin Square Assignment")
    print("="*70)
    
    csc_assignment = create_latin_square_assignment(csc_instances, SETTINGS, NUM_ANNOTATORS)
    multipico_assignment = create_latin_square_assignment(multipico_instances, SETTINGS, NUM_ANNOTATORS)
    
    print(f"  ✓ Assignment creato per {NUM_ANNOTATORS} annotatori")
    print(f"  ✓ Ogni annotatore valuterà {len(csc_instances)} istanze CSC")
    print(f"  ✓ Ogni annotatore valuterà {len(multipico_instances)} istanze MultiPICo")
    
    # Salva le associazioni in un JSON
    assignments_data = {
        "settings": SETTINGS,
        "num_annotators": NUM_ANNOTATORS,
        "csc": {
            f"annotator_{i+1}": {
                "unique_id_to_setting": csc_assignment[i],
                "setting_counts": {setting: list(csc_assignment[i].values()).count(setting) for setting in SETTINGS}
            }
            for i in range(NUM_ANNOTATORS)
        },
        "multipico": {
            f"annotator_{i+1}": {
                "unique_id_to_setting": multipico_assignment[i],
                "setting_counts": {setting: list(multipico_assignment[i].values()).count(setting) for setting in SETTINGS}
            }
            for i in range(NUM_ANNOTATORS)
        }
    }
    
    assignments_file = output_base / "latin_square_assignments.json"
    with open(assignments_file, 'w', encoding='utf-8') as f:
        json.dump(assignments_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Associazioni salvate in: {assignments_file.name}")
    
    # FASE 3: Genera file Excel per ogni annotatore
    print("\n" + "="*70)
    print("FASE 3: Generazione file Excel per annotatori")
    print("="*70)
    
    for annotator_id in range(NUM_ANNOTATORS):
        print(f"\n📝 Annotatore {annotator_id + 1}:")
        
        output_excel = output_base / f"annotator_{annotator_id + 1}_evaluation.xlsx"
        
        # Carica dati CSC
        print("  Caricamento CSC...")
        df_csc = load_data_for_annotator(
            csc_dir, 
            "CSC", 
            csc_instances, 
            csc_assignment[annotator_id]
        )
        
        # Carica dati MultiPICo
        print("  Caricamento MultiPICo...")
        df_multipico = load_data_for_annotator(
            multipico_dir, 
            "MultiPICo", 
            multipico_instances, 
            multipico_assignment[annotator_id]
        )
        
        # Salva Excel
        if df_csc is not None and df_multipico is not None:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                df_csc.to_excel(writer, sheet_name='CSC', index=False)
                df_multipico.to_excel(writer, sheet_name='MultiPICo', index=False)
            
            # Conta le label per CSC
            csc_label_counts = df_csc['aggregate_sarcasm_label'].value_counts().sort_index()
            # Conta le label per MultiPICo
            multipico_label_counts = df_multipico['aggregate_irony_label'].value_counts().sort_index()
            
            print(f"  ✓ File salvato: {output_excel.name}")
            print(f"    - Sheet CSC: {len(df_csc)} righe")
            print(f"      • Sarcasm label 0: {csc_label_counts.get(0)}")
            print(f"      • Sarcasm label 1: {csc_label_counts.get(1)}")
            print(f"    - Sheet MultiPICo: {len(df_multipico)} righe")
            print(f"      • Irony label 0: {multipico_label_counts.get(0)}")
            print(f"      • Irony label 1: {multipico_label_counts.get(1)}")
        else:
            print(f"  ⚠ Errore nel caricamento dati per annotatore {annotator_id + 1}")
    
    print("\n" + "="*70)
    print("🎉 Tutti i file Excel sono stati creati con successo!")
    print(f"📁 Directory output: {output_base.absolute()}")
    print("="*70)
        
if __name__ == "__main__":
    main()