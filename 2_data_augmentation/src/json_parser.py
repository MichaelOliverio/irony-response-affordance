"""
Utility per il parsing e la pulizia di JSON dalle risposte dei modelli LLM.
"""
import json
import re
from typing import Optional, Dict, Any
import json5


class JSONParser:
    """Classe per estrarre e parsare JSON dalle risposte dei modelli LLM."""

    """
    TODO: "turns": [{"u1": --> se qualcosa di questo è già presente allora non va reinserito
    TODO: se viene fatto un unico dizionario con chiavi ripetute (u1, u2) allora va splittato in più dizionari
    """

    @staticmethod
    def clean_message(message: str) -> str:
        """
        Pulisce un messaggio rimuovendo il prefisso 'Steve:' e i caratteri newline.
        
        Args:
            message: Il messaggio da pulire
            
        Returns:
            Il messaggio pulito
        """
        # Rimuove "Steve:" all'inizio del messaggio (case-insensitive)
        cleaned = re.sub(r'^Steve:\s*', '', message, flags=re.IGNORECASE)
        
        # Rimuove tutti i caratteri newline
        cleaned = cleaned.replace('\n', ' ')
        
        # Rimuove spazi multipli e fa strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    @staticmethod
    def has_repeated_utterances(messages: list) -> bool:
        """
        Controlla se ci sono utterance ripetute nel dialogo.
        
        Args:
            messages: Lista di dizionari con 'speaker' e 'content'
            
        Returns:
            True se ci sono messaggi ripetuti, False altrimenti
        """
        seen_contents = set()
        
        for msg in messages:
            content = msg['content'].strip().lower()
            if content in seen_contents:
                return True
            seen_contents.add(content)
        
        return False

    @staticmethod
    def extract_json_tag_content(text: str) -> Optional[str]:
        """Estrae il contenuto tra tag <JSON>...</JSON>"""
        match = re.search(r'<JSON\s*>(.*?)</JSON\s*>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
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


    @staticmethod
    def convert_to_valid_json(json_str: str) -> Optional[dict]:
        """Tenta di parsare il JSON, applicando prima le correzioni"""
        if not json_str:
            return None
        
        # Applica correzioni
        json_str = JSONParser.fix_json_string(json_str)
        
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

    @staticmethod
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
            print(" JSON non contiene 'turns'")
            return None
        
        turns = json_content["turns"]
        if not isinstance(turns, list) or len(turns) == 0:
            print(" 'turns' non è una lista valida")
            return None
        
        # Estrai tutti i messaggi da tutti i turns
        all_messages = []
        
        for turn_idx, turn in enumerate(turns):
            if not isinstance(turn, dict):
                print(f" Turn {turn_idx} non è un dizionario")
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
                        # Pulisci il messaggio (rimuovi "Steve:" e "\n")
                        cleaned_message = JSONParser.clean_message(message)
                        
                        if cleaned_message:  # Verifica che dopo la pulizia il messaggio non sia vuoto
                            turn_messages.append({
                                'speaker': speaker,
                                'content': cleaned_message
                            })
            
            all_messages.extend(turn_messages)
        
        if len(all_messages) == 0:
            print("Nessun messaggio trovato")
            return None
        
        # Controlla se ci sono utterance ripetute
        if JSONParser.has_repeated_utterances(all_messages):
            print("Dialogo non valido: trovate utterance ripetute")
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
            
            # Pulisci anche i messaggi originali
            all_messages[last_u1_idx]['content'] = JSONParser.clean_message(original_post)
            all_messages[last_u2_idx]['content'] = JSONParser.clean_message(original_reply)
            print(f" Sostituiti ultimi messaggi: u1 (idx={last_u1_idx}), u2 (idx={last_u2_idx})")
        else:
            # CSC: sostituisci solo l'ultimo u2
            if last_u2_idx is None:
                print("Non trovato u2 nei messaggi")
                return None
            
            # Pulisci anche il messaggio originale
            all_messages[last_u2_idx]['content'] = JSONParser.clean_message(original_reply)
            print(f" Sostituito ultimo messaggio u2 (idx={last_u2_idx})")
        
        # Ricontrolla se ci sono utterance ripetute dopo la sostituzione
        if JSONParser.has_repeated_utterances(all_messages):
            print("Dialogo non valido: trovate utterance ripetute dopo sostituzione dei messaggi originali")
            return None
        
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
        
        print(f" JSON validato con {len(normalized_turns)} turns normalizzati")
        
        return validated_json

    @staticmethod
    def multipico_parse(text: str, num_turns: int, post: str, reply: str) -> Optional[Dict[str, Any]]:
        """
        Parsing specifico per il dataset MultiPICo, che include post e reply esterni.
        
        Args:
            text: Testo di input contenente il blocco JSON
            num_turns: Numero di turni generati attesi
            post: Testo del post originale
            reply: Testo della reply originale
        """

        raw_json_content = JSONParser.extract_json_tag_content(text)
        if not raw_json_content:
            print("Nessun blocco <JSON> trovato.")
            return None

        # Converti a JSON valido
        json_content = JSONParser.convert_to_valid_json(raw_json_content)
        if json_content is None:
            print("Errore nel parsing del JSON.")
            return None
        
        # Valida il JSON
        validated_json_content = JSONParser.validate_json_content(
            json_content,
            num_turns, 
            post, 
            reply,
            "u1"  # Nome fittizio per u1
        )

        return validated_json_content
    
    @staticmethod
    def csc_parse(text: str, num_turns: int, reply: str) -> Optional[Dict[str, Any]]:
        """
        Parsing specifico per il dataset CSC, che include solo la reply esterna.
        
        Args:
            text: Testo di input contenente il blocco JSON
            num_turns: Numero di turni generati attesi
            reply: Testo della reply originale
        """

        raw_json_content = JSONParser.extract_json_tag_content(text)
        if not raw_json_content:
            print("Nessun blocco <JSON> trovato.")
            return None

        # Converti a JSON valido
        json_content = JSONParser.convert_to_valid_json(raw_json_content)
        if json_content is None:
            print("Errore nel parsing del JSON.")
            return None
        
        # Valida il JSON
        validated_json_content = JSONParser.validate_json_content(
            json_content,
            num_turns,
            None, 
            reply,
            "Steve"  # Nome fittizio per u1
        )

        return validated_json_content

    @staticmethod
    def parse(text: str, num_turns: int) -> Optional[Dict[str, Any]]:
        if not text:
            print("Testo vuoto.")
            return None

        # Estrai il blocco JSON
        match = re.search(r'<JSON\s*>(.*?)</JSON\s*>', text, re.DOTALL | re.IGNORECASE)
        if not match:
            print("Nessun blocco <JSON> trovato.")
            return None

        json_str = match.group(1).strip()

        # -------- PULIZIA JSON -------- #
        
        # 0. Normalizza virgolette curve in virgolette dritte
        json_str = json_str.replace('"', '"').replace('"', '"')
        json_str = json_str.replace(''', "'").replace(''', "'")
        
        # 1. Rimuove trailing commas illegali
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # 2. Fix per newline non escapate dentro le stringhe JSON
        # Trova tutte le stringhe tra virgolette e sostituisce \n con \\n
        def escape_newlines_in_strings(match):
            string_content = match.group(1)
            # Escape newlines, carriage returns, tabs
            string_content = string_content.replace('\n', '\\n')
            string_content = string_content.replace('\r', '\\r')
            string_content = string_content.replace('\t', '\\t')
            return f'"{string_content}"'
        
        # Pattern che matcha stringhe tra virgolette (anche multiline)
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)\"', escape_newlines_in_strings, json_str, flags=re.DOTALL)
        
        # 3. Rimuove chiavi duplicate o errate (es. last_turn, lastTurn)
        json_str = re.sub(r',\s*"last_?[tT]urn"\s*:\s*\{[^}]*\}', '', json_str)
        json_str = re.sub(r',\s*"last_?[tT]urn"\s*:\s*\[[^\]]*\]', '', json_str)

        # Parsing robusto con JSON5
        try:
            data = json5.loads(json_str)
        except Exception as e:
            print(f"Errore nel parsing JSON5: {e}")
            # Mostra solo le prime 500 caratteri per debug
            print(f"JSON estratto (primi 500 char):\n{json_str[:500]}...")
            return None

        # -------- VALIDAZIONI -------- #

        # Deve contenere solo la chiave "turns"
        if not isinstance(data, dict):
            print("Il JSON non è un oggetto.")
            return None

        # Accetta sia "turns" che altre chiavi comuni, ma rimuovi le extra
        if "turns" not in data:
            print(f'Chiave "turns" non trovata. Chiavi presenti: {list(data.keys())}')
            return None
        
        # Rimuovi chiavi extra mantenendo solo "turns"
        data = {"turns": data["turns"]}

        # turns deve essere una lista
        if not isinstance(data["turns"], list):
            print('"turns" non è una lista.')
            return None

        # Deve avere num_turns + 1 elementi
        expected = num_turns + 1
        actual = len(data["turns"])

        if actual != expected:
            print(f'  Numero di turni errato: trovato {actual}, atteso {expected}.')
            return None
        
        # Valida che ogni turn abbia u1 e u2
        for i, turn in enumerate(data["turns"]):
            if not isinstance(turn, dict):
                print(f'Turn {i} non è un oggetto.')
                return None
            if "u1" not in turn or "u2" not in turn:
                print(f'Turn {i} manca u1 o u2. Chiavi: {list(turn.keys())}')
                return None

        return data
