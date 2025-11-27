import json
import json5
import re
from pathlib import Path
from typing import Optional, Dict, Any


def extract_json_tag_content(text: str) -> Optional[str]:
    """Estrae il contenuto tra tag <JSON>...</JSON>"""
    match = re.search(r'<JSON\s*>(.*?)</JSON\s*>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def fix_json_string(json_str: str) -> str:
    """Applica correzioni comuni al JSON malformato"""
    # Rimuove trailing commas illegali
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    
    # Sostituisce newline letterali con spazi
    json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
    
    # Rimuove spazi multipli
    json_str = re.sub(r'\s+', ' ', json_str)
    
    # Strategia: splittiamo per "u1": e "u2": e processiamo ogni valore
    parts = []
    current_pos = 0
    
    # Pattern per trovare "u1": " o "u2": "
    pattern = r'"(u[12])":\s*"'
    
    for match in re.finditer(pattern, json_str):
        # Aggiungi tutto prima del match
        parts.append(json_str[current_pos:match.start()])
        
        # Aggiungi il match stesso
        key = match.group(1)
        parts.append(f'"{key}": "')
        
        # Ora trova dove finisce questo valore
        # Il valore finisce con " seguito da , o }
        value_start = match.end()
        i = value_start
        value_chars = []
        
        while i < len(json_str):
            char = json_str[i]
            
            # Gestisci escape esistenti
            if char == '\\' and i + 1 < len(json_str):
                value_chars.append(char)
                value_chars.append(json_str[i + 1])
                i += 2
                continue
            
            # Se troviamo una virgoletta, controlliamo se è la fine
            if char == '"':
                # Guarda il carattere dopo (saltando spazi)
                j = i + 1
                while j < len(json_str) and json_str[j] == ' ':
                    j += 1
                
                # Se dopo c'è virgola o parentesi graffa, è la fine
                if j >= len(json_str) or json_str[j] in ',}':
                    # Fine del valore
                    parts.append(''.join(value_chars))
                    parts.append('"')
                    current_pos = i + 1
                    break
                else:
                    # Virgoletta interna, escapala
                    value_chars.append('\\"')
            else:
                value_chars.append(char)
            
            i += 1
    
    # Aggiungi il resto
    parts.append(json_str[current_pos:])
    
    return ''.join(parts)


def convert_to_valid_json(json_str: str) -> Optional[dict]:
    """Tenta di parsare il JSON, applicando prima le correzioni"""
    if not json_str:
        return None
    
    # Applica correzioni
    json_str = fix_json_string(json_str)
    
    # Prova prima con JSON5 (più permissivo)
    try:
        data = json5.loads(json_str)
        return data
    except Exception as e:
        print(f"Errore nel parsing JSON5: {e}")
        
        # Prova con json standard
        try:
            data = json.loads(json_str)
            return data
        except Exception as e2:
            print(f"Errore anche con json standard: {e2}")
            return None
        
def validate_json_content(json_content: dict, num_turns_generated: int, original_post: str, original_reply: str, u1_name: str) -> Optional[dict]:
    """
    Valida e normalizza il contenuto JSON estratto.
    
    Args:
        json_content: Il JSON parsato da validare
        num_turns_generated: Numero di turni attesi (es. 3)
        original_post: Il post originale (ultimo u1)
        original_reply: La reply originale (ultimo u2)
    
    Returns:
        JSON validato e normalizzato, o None se non valido
    """
    if not json_content or "turns" not in json_content:
        print("❌ JSON non contiene 'turns'")
        return None
    
    turns = json_content["turns"]
    if not isinstance(turns, list) or len(turns) == 0:
        print("❌ 'turns' non è una lista valida")
        return None
    
    # Estrai tutti i messaggi da tutti i turns
    all_messages = []
    
    for turn_idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            print(f"❌ Turn {turn_idx} non è un dizionario")
            continue
        
        # Estrai tutti i messaggi u1 e u2 da questo turn
        # I messaggi potrebbero essere multipli nello stesso oggetto
        turn_messages = []
        
        for key in sorted(turn.keys()):
            if key.startswith(u1_name) or key.startswith('u2'):
                # Determina se è u1 o u2
                speaker = 'u1' if key.startswith(u1_name) else 'u2'
                message = turn[key]
                
                if isinstance(message, str) and message.strip():
                    turn_messages.append({
                        'speaker': speaker,
                        'content': message.strip()
                    })
        
        all_messages.extend(turn_messages)
    
    if len(all_messages) == 0:
        print("Nessun messaggio trovato")
        return None
    
    print(f"Trovati {len(all_messages)} messaggi totali")
    
    # Calcola il numero atteso di messaggi: num_turns * 2 + 2
    expected_messages = num_turns_generated * 2 + 2
    
    print(f"Messaggi attesi: {expected_messages} (num_turns={num_turns_generated})")
    
    # Prendi gli ultimi N messaggi (dal fondo)
    if len(all_messages) > expected_messages:
        print(f"Troppi messaggi ({len(all_messages)}), prendo gli ultimi {expected_messages}")
        all_messages = all_messages[-expected_messages:]
    elif len(all_messages) < expected_messages:
        print(f"Pochi messaggi ({len(all_messages)}), attesi {expected_messages}")
        return None
    
    # Sostituisci gli ultimi due messaggi con i valori originali
    # Ultimo u1 = original_post
    # Ultimo u2 = original_reply
    
    # Trova l'indice dell'ultimo u1 e dell'ultimo u2
    last_u1_idx = None
    last_u2_idx = None
    
    for idx in range(len(all_messages) - 1, -1, -1):
        if all_messages[idx]['speaker'] == 'u1' and last_u1_idx is None:
            last_u1_idx = idx
        if all_messages[idx]['speaker'] == 'u2' and last_u2_idx is None:
            last_u2_idx = idx
        
        if last_u1_idx is not None and last_u2_idx is not None:
            break
    
    # Sostituisci i contenuti in base al dataset
    if original_post is not None:
        # MultiPICo: sostituisci sia l'ultimo u1 che l'ultimo u2
        if last_u1_idx is None or last_u2_idx is None:
            print("Non trovati sia u1 che u2 nei messaggi")
            return None
        
        all_messages[last_u1_idx]['content'] = original_post
        all_messages[last_u2_idx]['content'] = original_reply
        print(f"✅ Sostituiti ultimi messaggi: u1 (idx={last_u1_idx}), u2 (idx={last_u2_idx})")
    else:
        # CSC: sostituisci solo l'ultimo u2
        if last_u2_idx is None:
            print("Non trovato u2 nei messaggi")
            return None
        
        all_messages[last_u2_idx]['content'] = original_reply
        print(f"✅ Sostituito ultimo messaggio u2 (idx={last_u2_idx})")
    
    # Ricostruisci il JSON normalizzato con il formato corretto
    # Formato: un oggetto per ogni coppia (u1, u2)
    normalized_turns = []
    
    i = 0
    while i < len(all_messages):
        turn = {}
        
        # Aggiungi u1 se disponibile
        if i < len(all_messages) and all_messages[i]['speaker'] == 'u1':
            turn['u1'] = all_messages[i]['content']
            i += 1
        
        # Aggiungi u2 se disponibile
        if i < len(all_messages) and all_messages[i]['speaker'] == 'u2':
            turn['u2'] = all_messages[i]['content']
            i += 1
        
        # Se abbiamo almeno uno dei due, aggiungi il turn
        if turn:
            normalized_turns.append(turn)
    
    # Se l'ultimo turn ha solo u1 o solo u2, potrebbe essere incompleto
    # ma lo teniamo comunque perché contiene post o reply originale
    
    validated_json = {
        "turns": normalized_turns
    }
    
    print(f"✅ JSON validato con {len(normalized_turns)} turns normalizzati")
    
    return validated_json

def main():
    """Funzione principale per riprocessare i file JSON"""
    output_dir = Path("output")
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print("❌ Nessun file JSON trovato nella directory output/")
        return
    
    # Statistiche globali
    total_stats = {
        'total_entries': 0,
        'entries_with_errors': 0,
        'entries_reprocessed': 0,
        'entries_fixed': 0,
        'entries_failed': 0
    }
    
    for json_file in json_files:
        print(f"\n{'='*70}")
        print(f"📂 Processing file: {json_file}")
        print(f"{'='*70}\n")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"📊 Number of entries in file: {len(data)}")
        total_stats['total_entries'] += len(data)
        
        entries_modified = []
        
        for entry_idx, entry in enumerate(data):
            unique_id = entry.get("unique_id", None)
            post = entry.get("post", None)
            reply = entry.get("reply", None)
            num_turns_generated = entry.get("num_turns_generated", None)
            dialogue = entry.get("dialogue", None)
            invalid_model_outputs = entry.get("invalid_model_outputs", None)

            # Skip se mancano campi essenziali
            if unique_id is None or post is None or reply is None:
                continue
            
            # Skip se non c'è dialogo o non ha errori
            if dialogue is None:
                continue
            
            if not isinstance(dialogue, dict) or "error" not in dialogue:
                continue

            print(f"\n🔄 Processing entry {entry_idx + 1}/{len(data)}: {unique_id}")
            total_stats['entries_with_errors'] += 1

            # Verifica se ci sono output invalidi da riprocessare
            if not invalid_model_outputs or len(invalid_model_outputs) == 0:
                print("⚠️  No invalid model outputs to reprocess.")
                continue

            total_stats['entries_reprocessed'] += 1
            
            # Prova a riprocessare ogni output invalido
            best_result = None
            
            for output_idx, invalid_output in enumerate(invalid_model_outputs):
                print(f"\n   📝 Attempting to parse invalid output {output_idx + 1}/{len(invalid_model_outputs)}...")
                
                initial_output = """<JSON>{"turns": [{"u1":\""""
                output = f"{initial_output}{invalid_output}"
                
                # Estrai il JSON
                raw_json_content = extract_json_tag_content(output)
                if not raw_json_content:
                    print(f"   ❌ Failed to extract JSON from output {output_idx + 1}")
                    continue

                # Converti a JSON valido
                json_content = convert_to_valid_json(raw_json_content)
                if json_content is None:
                    print(f"   ❌ Failed to parse JSON from output {output_idx + 1}")
                    continue
                
                print(f"   ✅ JSON parsed successfully from output {output_idx + 1}")

                # Valida il JSON
                validated_json_content = validate_json_content(
                    json_content,
                    num_turns_generated, 
                    None, 
                    reply,
                    "Steve"
                )
                
                if validated_json_content is not None:
                    print(f"   ✅ JSON validated successfully from output {output_idx + 1}!")
                    best_result = validated_json_content
                    break  # Usa il primo output valido
                else:
                    print(f"   ❌ Validation failed for output {output_idx + 1}")
            
            # Se abbiamo trovato un risultato valido, aggiorna l'entry
            if best_result is not None:
                entry['dialogue'] = best_result
                entry['invalid_inferences'] = 0  # Reset del contatore errori
                entries_modified.append(unique_id)
                total_stats['entries_fixed'] += 1
                print(f"\n✅ Entry {unique_id} fixed!")
            else:
                total_stats['entries_failed'] += 1
                print(f"\n❌ Could not fix entry {unique_id}")
        
        # Salva il file modificato
        if entries_modified:
            print(f"\n💾 Saving modified file with {len(entries_modified)} fixed entries...")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ File saved: {json_file}")
            print(f"   Fixed entries: {', '.join(entries_modified)}")
        else:
            print(f"\n⚠️  No entries were modified in this file")
        
    
    # Stampa statistiche finali
    print(f"\n{'='*70}")
    print(f"📊 FINAL STATISTICS")
    print(f"{'='*70}")
    print(f"Total entries processed: {total_stats['total_entries']}")
    print(f"Entries with errors: {total_stats['entries_with_errors']}")
    print(f"Entries reprocessed: {total_stats['entries_reprocessed']}")
    print(f"Entries fixed: {total_stats['entries_fixed']}")
    print(f"Entries failed: {total_stats['entries_failed']}")
    print(f"\n✅ Processing complete!")


if __name__ == "__main__":
    main()