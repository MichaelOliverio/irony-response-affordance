"""
Utility per il parsing e la pulizia di JSON dalle risposte dei modelli LLM.
"""
import json
import re
from typing import Optional, Dict, Any


class JSONParser:
    """Classe per estrarre e parsare JSON dalle risposte dei modelli LLM."""
    
    @staticmethod
    def extract_and_parse(text: str) -> Optional[Dict[str, Any]]:
        """
        Estrae e ripara il JSON tra <JSON> e </JSON> da un testo.
        
        Args:
            text: Testo contenente il JSON da estrarre
        
        Returns:
            Dizionario parsato o None se il parsing fallisce
        """
        if not text:
            return None

        # Trova il primo blocco <JSON> ... </JSON>
        match = re.search(r'<JSON\s*>(.*?)</JSON\s*>', text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None

        json_str = match.group(1).strip()

        # Applica escape alle stringhe
        json_str = JSONParser._escape_all_strings(json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON: {e}")
            return None
    
    @staticmethod
    def _escape_json_string(s: str) -> str:
        """Applica escape ai caratteri speciali in una stringa JSON."""
        s = s.replace('\\', '\\\\')   # escape backslash
        s = s.replace('"', '\\"')     # escape double quotes
        s = s.replace('\n', '\\n')    # escape newlines
        s = s.replace('\r', '\\r')    # escape carriage return
        return s
    
    @staticmethod
    def _escape_all_strings(json_text: str) -> str:
        """
        Trova tutte le stringhe dentro virgolette e applica l'escape.
        
        Args:
            json_text: Testo JSON da processare
        
        Returns:
            Testo JSON con stringhe escapate
        """
        def replacer(match):
            inner = match.group(1)
            return f'"{JSONParser._escape_json_string(inner)}"'
        
        return re.sub(r'"(.*?)"', replacer, json_text, flags=re.DOTALL)
