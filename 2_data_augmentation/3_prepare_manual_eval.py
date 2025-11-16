import json
import re
import pandas as pd
from pathlib import Path

output_dir = Path("output")
json_files = list(output_dir.glob("*.json"))

# Regex per estrarre modello e numero turning dal nome del file
file_pattern = re.compile(r"MultiPICo_(.+)_num_gen_turn_(\d+)\.json")

# Dizionario: key = (model, num_turns), value = lista dei record
grouped = {}

for json_file in json_files:
    m = file_pattern.search(json_file.name)
    if not m:
        continue

    model = m.group(1)
    num_turns = int(m.group(2))

    # Carica file
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filtra righe con dialogue.turns
    filtered_rows = [
        row for row in data
        if isinstance(row.get("dialogue"), dict)
        and "turns" in row["dialogue"]
    ]

    # Aggiungi al gruppo
    key = (model, num_turns)
    grouped.setdefault(key, []).extend(filtered_rows)

def format_dialogue(turns):
    """
    turns è una lista di dict come:
    [{'u1': '...', 'u2': '...'}, {...}]
    """
    formatted_blocks = []
    for block in turns:
        lines = []
        for speaker, text in block.items():
            lines.append(f"{speaker}: {text}")
        formatted_blocks.append("\n".join(lines))
    return "\n-----------------------\n".join(formatted_blocks)

def main():
    # Crea un Excel separato per ogni combinazione (model, num_turns)
    output_base = Path("manual_eval_files")
    output_base.mkdir(exist_ok=True)

    for (model, num_turns), rows in grouped.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)

        df = df[["unique_id", "dataset_name", "post", "reply", "dialogue"]]

        df['dialogue'] = df['dialogue'].apply(lambda d: format_dialogue(d['turns']))

        safe_model = re.sub(r"[^A-Za-z0-9_]+", "_", model)
        output_excel = output_base / f"{safe_model}_turns_{num_turns}.xlsx"

        # Salva il file (usa engine predefinito: xlsxwriter o openpyxl)
        df.to_excel(output_excel, index=False)

        print(f"Creato: {output_excel}")
        
if __name__ == "__main__":
    main()