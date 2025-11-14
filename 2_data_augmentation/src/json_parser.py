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

        # Rimuove trailing commas illegali
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        # Parsing robusto con JSON5
        try:
            data = json5.loads(json_str)
        except Exception as e:
            print("Errore nel parsing JSON5:", e)
            print("JSON estratto:\n", json_str)
            return None

        # -------- VALIDAZIONI -------- #

        # Deve contenere solo la chiave "turns"
        if not isinstance(data, dict):
            print("Il JSON non è un oggetto.")
            return None

        keys = list(data.keys())
        if keys != ["turns"]:
            print(f"Chiavi JSON errate: {keys}, attesa solo ['turns']")
            return None

        # turns deve essere una lista
        if not isinstance(data["turns"], list):
            print('"turns" non è una lista.')
            return None

        # Deve avere num_turns + 1 elementi
        expected = num_turns + 1
        actual = len(data["turns"])

        if actual != expected:
            print(f'Numero di turni errato: trovato {actual}, atteso {expected}.')
            return None

        return data
