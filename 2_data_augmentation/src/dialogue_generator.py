"""
Generatore di dialoghi multi-turn usando modelli LLM.
"""
import re
import pandas as pd
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from .llm_handler import LLMHandler
from .json_parser import JSONParser
from .config import DATA_AUGMENTATION_HYPERPARAMS, DATA_AUGMENTATION_MODELS, DATA_AUGMENTATION_OUTPUT_DIR

class DialogueGenerator:
    """Classe per generare dialoghi multi-turn da coppie post-reply."""
    @staticmethod
    def generate_batch_output(
        pipe, 
        tokenizer, 
        turns: List[Tuple[str, str]], 
    ) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
        """
        Genera output in batch per una lista di coppie di messaggi.
        
        Args:
            pipe: Pipeline del modello di generazione
            tokenizer: Tokenizer associato al modello
            turns: Lista di tuple (post, reply)
        
        Returns:
            Lista di tuple (prompt, model_output, dialogue_dict)
        """
        prompts = []
        for post, reply in turns:
            prompt = DialogueGenerator._create_prompt(post, reply)
            prompts.append(prompt)

        # Genera i risultati dal modello
        outputs = pipe(
            prompts,
            max_new_tokens=DATA_AUGMENTATION_HYPERPARAMS['max_new_tokens'],
            do_sample=DATA_AUGMENTATION_HYPERPARAMS['temperature'] > 0,
            temperature=DATA_AUGMENTATION_HYPERPARAMS['temperature'],
            return_full_text=False,
            batch_size=DATA_AUGMENTATION_HYPERPARAMS['batch_size'],
            pad_token_id=tokenizer.eos_token_id
        )

        results = []
        for prompt, out in zip(prompts, outputs):
            entry = out[0]
            model_output = entry['generated_text'].strip()
            
            # Ricostruisce l'output completo aggiungendo il prefisso iniziale
            initial_output = """<JSON>{"turns": [{"u1":\""""
            dialogue = JSONParser.extract_and_parse(f"{initial_output}{model_output}")
            
            results.append((prompt, model_output, dialogue))

        return results
    
    @staticmethod
    def _create_prompt(post: str, reply: str) -> str:
        """
        Crea il prompt per la generazione del dialogo.
        
        Args:
            post: Messaggio iniziale (u1)
            reply: Risposta (u2)
        
        Returns:
            Prompt formattato
        """
        return f"""
            Given this pair of messages extracted from social media:

            u1: {post}
            u2: {reply}

            Generate the 3 previous dialogue turns (each turn consisting of a message from u1 and a message from u2) that logically lead up to the given messages. Make sure to:
            - Maintain the original writing style of both u1 e u2.
            - Preserve the sentiment and tone of the conversation.
            - Keep the dialogue coherent and consistent with the context of the given messages.

            Return the result in the following JSON format, including both the previous 3 turns and the given messages as the last turn, enclosed in <JSON> tags:

            <JSON>
            {{
                "turns": [
                    {{
                        "u1": "First message from user1 in the earliest turn",
                        "u2": "First message from user2 in response"
                    }},
                    {{
                        "u1": "Second message from user1",
                        "u2": "Second message from user2"
                    }},
                    {{
                        "u1": "Third message from user1",
                        "u2": "Third message from user2"
                    }},
                    {{
                        "u1": "{post}",
                        "u2": "{reply}"
                    }}
                ]
            }}
            </JSON>

            Response:
            <JSON>
            {{
                "turns": [
                    {{
                        "u1": "
            """
    
    @staticmethod
    def generate_dialogues_for_dataset(
        dataset_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Genera dialoghi per un intero dataset usando più modelli.
        
        Args:
            dataset_df: DataFrame con colonne 'post', 'reply', 'unique_id'
        
        Returns:
            Dizionario {model_name: output_file_path}
        """        
        output_files = {}
        
        for model_name in DATA_AUGMENTATION_MODELS:
            print(f"\n{'='*60}")
            print(f"Processing model: {model_name}")
            print(f"{'='*60}")
            
            pipe, model, tokenizer = LLMHandler.load_model(model_name)

            output_file = os.path.join(
                DATA_AUGMENTATION_OUTPUT_DIR, 
                f'augmented_{model_name.replace("/", "_")}.json'
            )
            output_rows = []

            # Carica dati già esistenti, se presenti
            existing_ids = set()
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    existing_ids = set([row['unique_id'] for row in existing_data])
                    output_rows.extend(existing_data)
                    print(f"Trovati {len(existing_ids)} record già processati in {output_file}")
                except Exception as e:
                    print(f"Errore nel caricamento del JSON esistente: {e}")

            # Filtra dataset per saltare già processati
            dataset_to_process = dataset_df[~dataset_df['unique_id'].isin(existing_ids)]
            print(f"Record da processare: {len(dataset_to_process)}")

            if len(dataset_to_process) == 0:
                print("✓ Tutti i record sono già stati processati")
                LLMHandler.destroy_model(model, tokenizer)
                output_files[model_name] = output_file
                continue

            # Processa i batch
            for start in tqdm(range(0, len(dataset_to_process), DATA_AUGMENTATION_HYPERPARAMS['batch_size']), 
                            desc=f"Running {model_name}"):
                batch_df = dataset_to_process.iloc[start:start+DATA_AUGMENTATION_HYPERPARAMS['batch_size']]

                # Prepara post e reply
                post = batch_df['post'].tolist()
                #post = [re.sub(r'@USER', '', p) for p in post]

                reply = batch_df['reply'].tolist()
                #reply = [re.sub(r'@USER', '', r) for r in reply]

                turns = list(zip(post, reply))

                # Genera output
                results = DialogueGenerator.generate_batch_output(
                    pipe, tokenizer, turns
                )

                # Gestisce retry per inferenze fallite
                cleaned_results = []
                for i, (prompt, model_output, dialogue) in enumerate(results):
                    invalid_inferences = 0
                    while dialogue is None and invalid_inferences < DATA_AUGMENTATION_HYPERPARAMS['max_retries']:
                        invalid_inferences += 1
                        print(f"Retry {invalid_inferences}/{DATA_AUGMENTATION_HYPERPARAMS['max_retries']} per entry {batch_df.iloc[i]['unique_id']}")
                        prompt, model_output, dialogue = DialogueGenerator.generate_batch_output(
                            pipe, tokenizer, [turns[i]]
                        )[0]

                    if invalid_inferences >= DATA_AUGMENTATION_HYPERPARAMS['max_retries']:
                        print(f"Max retries raggiunto per entry {batch_df.iloc[i]['unique_id']}")
                        dialogue = {"error": "Max retries reached"}
                        model_output = {"error": "Max retries reached"}

                    cleaned_results.append((prompt, model_output, dialogue, invalid_inferences))

                # Salva risultati
                for row, (prompt, model_output, dialogue, invalid_inferences) in zip(
                    batch_df.itertuples(), cleaned_results
                ):
                    enriched = row._asdict()
                    enriched['prompt'] = prompt
                    enriched['model_output'] = model_output
                    enriched['dialogue'] = dialogue
                    enriched['invalid_inferences'] = invalid_inferences
                    enriched['model_name'] = model_name
                    output_rows.append(enriched)

                # Salva ad ogni batch
                pd.DataFrame(output_rows).to_json(
                    output_file,
                    orient='records',
                    indent=2,
                    force_ascii=False
                )

            print(f"Completato! Salvato in: {output_file}")
            print(f"   Totale record: {len(output_rows)}")
            
            LLMHandler.destroy_model(model, tokenizer)
            output_files[model_name] = output_file

        return output_files
