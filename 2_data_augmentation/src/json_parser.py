"""
Utility per il parsing e la pulizia di JSON dalle risposte dei modelli LLM.
"""
import json
import re
from typing import Optional, Dict, Any
import json5


class JSONParser:
    """Classe per estrarre e parsare JSON dalle risposte dei modelli LLM."""
    

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
            print(f'⚠️  Numero di turni errato: trovato {actual}, atteso {expected}.')
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
