import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

def load_all_files(directory: Path) -> Dict[str, List[dict]]:
    """
    Carica tutti i file JSON da una directory.
    
    Returns:
        Dict con filename come chiave e lista di dialoghi come valore
    """
    files_data = {}
    for json_file in directory.glob("*.json"):
        if "_filtered" not in json_file.name:
            with open(json_file, 'r', encoding='utf-8') as f:
                files_data[json_file.name] = json.load(f)
    return files_data

def find_valid_unique_ids(files_data: Dict[str, List[dict]]) -> Set[str]:
    """
    Trova i unique_id che hanno 'turns' in TUTTI i file.
    
    Args:
        files_data: Dict con i dati di tutti i file
        
    Returns:
        Set di unique_id validi (presenti con 'turns' in tutti i file)
    """
    # Per ogni unique_id, conta in quanti file ha 'turns' e in quanti è presente
    id_with_turns = defaultdict(int)
    id_total_count = defaultdict(int)
    
    for filename, dialogues in files_data.items():
        for dialogue in dialogues:
            unique_id = dialogue.get('unique_id')
            if unique_id:
                id_total_count[unique_id] += 1
                
                # Verifica se ha 'turns'
                if ('dialogue' in dialogue and 
                    isinstance(dialogue['dialogue'], dict) and 
                    'turns' in dialogue['dialogue'] and
                    dialogue['dialogue']['turns']):  # Verifica che non sia vuoto
                    id_with_turns[unique_id] += 1
    
    # Un unique_id è valido se in TUTTI i file in cui appare ha 'turns'
    valid_ids = set()
    for unique_id, count in id_total_count.items():
        if id_with_turns[unique_id] == count:
            valid_ids.add(unique_id)
    
    return valid_ids

def filter_and_save_files(directory: Path, files_data: Dict[str, List[dict]], valid_ids: Set[str]):
    """
    Filtra i file mantenendo solo i unique_id validi e salva i risultati.
    
    Args:
        directory: Directory dei file originali
        files_data: Dict con i dati di tutti i file
        valid_ids: Set di unique_id validi
    """
    # Crea la directory di output
    output_dir = directory.parent.parent / "cleaned_dialogues" / directory.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, dialogues in files_data.items():
        # Filtra i dialoghi
        filtered_dialogues = [
            d for d in dialogues 
            if d.get('unique_id') in valid_ids
        ]
        
        # Salva il file filtrato
        output_file = output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_dialogues, f, indent=2, ensure_ascii=False)
        
        print(f"  {filename}:")
        print(f"    - Originali: {len(dialogues)}")
        print(f"    - Filtrati: {len(filtered_dialogues)}")
        print(f"    - Rimossi: {len(dialogues) - len(filtered_dialogues)}")

def process_dataset(dataset_dir: Path, dataset_name: str):
    """
    Processa un intero dataset (CSC o MultiPICo).
    
    Args:
        dataset_dir: Directory del dataset
        dataset_name: Nome del dataset (per logging)
    """
    print(f"\n{'='*70}")
    print(f"Processando dataset: {dataset_name}")
    print(f"Directory: {dataset_dir}")
    print(f"{'='*70}\n")
    
    # 1. Carica tutti i file
    print("📂 Caricamento file...")
    files_data = load_all_files(dataset_dir)
    print(f"   Caricati {len(files_data)} file")
    
    # 2. Trova gli unique_id validi (con 'turns' in tutti i file)
    print("\n🔍 Analisi unique_id...")
    valid_ids = find_valid_unique_ids(files_data)
    
    # Statistiche
    all_ids = set()
    ids_without_turns = defaultdict(list)
    
    for filename, dialogues in files_data.items():
        for dialogue in dialogues:
            unique_id = dialogue.get('unique_id')
            if unique_id:
                all_ids.add(unique_id)
                
                # Verifica se manca 'turns'
                if not ('dialogue' in dialogue and 
                       isinstance(dialogue['dialogue'], dict) and 
                       'turns' in dialogue['dialogue'] and
                       dialogue['dialogue']['turns']):
                    ids_without_turns[unique_id].append(filename)
    
    invalid_ids = all_ids - valid_ids
    
    print(f"   Unique ID totali: {len(all_ids)}")
    print(f"   Unique ID validi (con 'turns' in tutti i file): {len(valid_ids)}")
    print(f"   Unique ID da rimuovere: {len(invalid_ids)}")
    
    if invalid_ids:
        print(f"\n   ⚠ Esempi di ID rimossi e i file in cui mancano 'turns':")
        for i, (uid, files) in enumerate(sorted(ids_without_turns.items())[:5], 1):
            print(f"      {i}. {uid}")
            for f in files:
                print(f"         - {f}")
    
    # 3. Filtra e salva i file
    print(f"\n💾 Salvataggio file filtrati...")
    filter_and_save_files(dataset_dir, files_data, valid_ids)
    
    print(f"\n✅ {dataset_name} completato!")

def main():
    # Directory base
    base_dir = Path(__file__).parent / "human_annotations" / "raw_dialogues"
    
    # Directory per CSC e MultiPICo
    csc_dir = base_dir / "csc_dialogues"
    multipico_dir = base_dir / "multipico_dialogues"
    
    print("🚀 Avvio filtro dialoghi\n")
    
    # Processa CSC
    if csc_dir.exists():
        process_dataset(csc_dir, "CSC")
    else:
        print(f"⚠ Directory CSC non trovata: {csc_dir}")
    
    # Processa MultiPICo
    if multipico_dir.exists():
        process_dataset(multipico_dir, "MultiPICo")
    else:
        print(f"⚠ Directory MultiPICo non trovata: {multipico_dir}")
    
    print("\n" + "="*70)
    print("🎉 Processo completato!")
    print("="*70)

if __name__ == "__main__":
    main()

