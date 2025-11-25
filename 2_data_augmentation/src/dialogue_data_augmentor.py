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
from .config import DATA_AUGMENTATION_HYPERPARAMS, DATA_AUGMENTATION_MODELS, PROMPT_MAPPING, NUM_MAX_TURNS

class DialogueDataAugmentor:
    """Classe per generare dialoghi multi-turn da coppie post-reply."""
    @staticmethod
    def generate_batch_output(
        pipe, 
        tokenizer, 
        turns: List[Tuple[str, str]], 
        dataset_name: str,
        num_turns: int
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
        posts = []
        replies = []
        for post, reply, relation in turns:
            posts.append(post)
            replies.append(reply)
            
            if dataset_name == "MultiPICo":
                prompt = DialogueDataAugmentor.multipico_prompt(post, reply, num_turns)
            elif dataset_name == "CSC":
                prompt = DialogueDataAugmentor.csc_prompt(post, reply, num_turns, relation)
            else:
                raise ValueError(f"Dataset non supportato: {dataset_name}")
            
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
        i = 0
        for prompt, out in zip(prompts, outputs):
            entry = out[0]
            model_output = entry['generated_text'].strip()
            
            # Ricostruisce l'output completo aggiungendo il prefisso iniziale
            initial_output = """<JSON>{"turns": [{"u1":\""""
            if dataset_name == "MultiPICo":
                dialogue = JSONParser.multipico_parse(f"{initial_output}{model_output}", num_turns, posts[i], replies[i])
            elif dataset_name == "CSC":
                dialogue = JSONParser.csc_parse(f"{initial_output}{model_output}", num_turns, replies[i])
            
            results.append((prompt, model_output, dialogue))
            i += 1
        return results
    
    @staticmethod
    def multipico_prompt(post, reply, num_turns) -> str:
        initial_prompt = f"""
        Given this pair of messages extracted from social media between the users u1 and user u2:

        u1: {post}
        u2: {reply}

        Generate the {num_turns} previous dialogue turns (each turn consisting of a message from u1 and a message from u2) that logically lead up to the given messages. Make sure to:
        - Maintain the original writing style of both u1 and u2.
        - Preserve the sentiment and tone of the conversation.
        - Keep the dialogue coherent and consistent with the context of the given messages.
        - **Do not** change the tone of the conversation, and ensure that the added turns blend well with the exchange between u1 and u2.
        - **Do not** produce duplicated or defective messages across the turns.
        
        Return ONLY a JSON object containing:
        - A "turns" array with exactly {num_turns + 1} turn objects (the {num_turns} previous turns plus the given messages as the last turn)
        - Each turn object must have this structure:
        {{
            "u1": "<message from u1>",
            "u2": "<message from u2>"
        }}

        Format:
        <JSON>
        {{
            "turns": ["""

        # turni
        turni_prompt = ""
        for i in range(num_turns):
            turni_prompt += f"""
            {{
                "u1": "{PROMPT_MAPPING[i+1]} message from u1",
                "u2": "{PROMPT_MAPPING[i+1]} message from u2"
            }},"""
        
        # Ultimo turno (quello dato)
        turni_prompt += f"""
        {{
            "u1": "{post}",
            "u2": "{reply}"
        }}"""

        final_prompt = f"""
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

        prompt = initial_prompt + turni_prompt + final_prompt

        return prompt
    
    @staticmethod
    def csc_prompt(context, reply, num_turns, relation) -> str:
        initial_prompt = f"""
        Given the context and the reply of a user u2 to u1 (whose name is Steve and your relation with him is {relation}):

        context: {context}
        u2: {reply}

        Generate the {num_turns} previous dialogue turns (each turn = one message from u1 followed by one from u2) plus the u1 message that directly precedes the final given u2 line. Make sure to:
        - Maintain the original writing style of u2.
        - Preserve the sentiment and tone of the conversation.
        - Keep the dialogue coherent and consistent with the context of the given messages.
        - **Do not** change the tone of the conversation, and ensure that the added turns blend well with the exchange between u1 and u2.
        - **Do not** produce duplicated or defective messages across the turns.

        Return ONLY a JSON object containing:
        - A "turns" array with exactly {num_turns + 1} turn objects (the {num_turns} previous turns plus the given messages as the last turn)
        - Each turn object must have this structure:
        {{
            "u1": "<message from u1>",
            "u2": "<message from u2>"
        }}

        Format:
        <JSON>
        {{
            "turns": ["""

        # turni
        turni_prompt = ""
        for i in range(num_turns):
            turni_prompt += f"""
            {{
                "u1": "{PROMPT_MAPPING[i+1]} message from u1",
                "u2": "{PROMPT_MAPPING[i+1]} message from u2"
            }},"""
            
        # Ultimo turno
        turni_prompt += f"""
        {{
            "u1": "Message from u1 that precedes the given u2",
            "u2": "{reply}"
        }}"""

        final_prompt = f"""
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

        prompt = initial_prompt + turni_prompt + final_prompt

        return prompt
    
    @staticmethod
    def generate_dialogues_for_dataset(
        dataset_df: pd.DataFrame,
        dataset_name: str,
    ) -> Dict[str, str]:
        for model_name in DATA_AUGMENTATION_MODELS:
            print(f"Processing model: {model_name}")
            pipe, model, tokenizer = LLMHandler.load_model(model_name)
            print("Model and tokenizer loaded.")
        
            for i in range(NUM_MAX_TURNS):
                num_turn = i + 1

                output_file = f'output/{dataset_name}_{model_name.replace("/", "_")}_n_turn_{num_turn}.json'
                output_rows = []

                # --- carica dati già esistenti, se presenti ---
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                        existing_ids = set([row['unique_id'] for row in existing_data])
                        print(f"Found {len(existing_ids)} already processed entries in {output_file}")
                        # aggiungi i dati esistenti a output_rows
                        output_rows.extend(existing_data)
                    except Exception as e:
                        print(f"Error loading existing JSON: {e}")
                        existing_ids = set()
                else:
                    existing_ids = set()

                # filtra dataset_df per saltare già processati
                dataset_df = dataset_df[~dataset_df['unique_id'].isin(existing_ids)]
                print(f"{len(dataset_df)} new entries to process")

                for start in tqdm(range(0, len(dataset_df), DATA_AUGMENTATION_HYPERPARAMS['batch_size']), desc=f"Running {model_name}"):
                    batch_df = dataset_df.iloc[start:start+DATA_AUGMENTATION_HYPERPARAMS['batch_size']]

                    post = batch_df['post'].tolist()
                    #post = [re.sub(r'@USER', '', p) for p in post]

                    reply = batch_df['reply'].tolist()
                    #reply = [re.sub(r'@USER', '', r) for r in reply]

                    if dataset_name == "CSC":
                        relation = batch_df['annotations'].apply(lambda x: x['speaker_annotation']['interlocutor_relationship']).tolist()
                    else:
                        relation = [None] * len(post)

                    turn = list(zip(post, reply, relation))

                    results = DialogueDataAugmentor.generate_batch_output(pipe, tokenizer, turn, dataset_name, num_turn)

                    cleaned_results = []
                    for i, (prompt, model_output, dialogue) in enumerate(results):
                        invalid_inferences = 0
                        invalid_model_outputs = []
                        print(f"Processing entry {batch_df.iloc[i]['unique_id']}")
                        while dialogue is None and invalid_inferences < DATA_AUGMENTATION_HYPERPARAMS['max_retries']:
                            print(f"Retrying entry {batch_df.iloc[i]['unique_id']} (attempt {invalid_inferences + 1})")
                            invalid_inferences += 1
                            invalid_model_outputs.append(model_output)
                            prompt, model_output, dialogue = DialogueDataAugmentor.generate_batch_output(pipe, tokenizer, [turn[i]], dataset_name, num_turn)[0]
                        print(f"Finished processing entry {batch_df.iloc[i]['unique_id']} with {invalid_inferences} invalid inferences.")

                        if invalid_inferences >= DATA_AUGMENTATION_HYPERPARAMS['max_retries']:
                            print(f"Warning: Max retries reached for entry {batch_df.iloc[i]['unique_id']}. Saving last output anyway.")
                            dialogue = {"error": "Max retries reached"}
                            model_output = {"error": "Max retries reached"}

                        cleaned_results.append((prompt, model_output, dialogue, num_turn, invalid_inferences, invalid_model_outputs))

                    for row, (prompt, model_output, dialogue, num_turns_generated, invalid_inferences, invalid_model_outputs) in zip(batch_df.itertuples(), cleaned_results):
                        enriched = row._asdict()  # Nota: _asdict() con underscore per pandas named tuples
                        enriched['prompt'] = prompt
                        enriched['model_output'] = model_output
                        enriched['dialogue'] = dialogue
                        enriched['num_turns_generated'] = num_turns_generated
                        enriched['invalid_model_outputs'] = invalid_model_outputs
                        enriched['invalid_inferences'] = invalid_inferences
                        output_rows.append(enriched)

                    # salva ad ogni batch
                    pd.DataFrame(output_rows).to_json(
                        output_file,
                        orient='records',
                        indent=2,
                        force_ascii=False
                    )
                    print(f"Batch {start//DATA_AUGMENTATION_HYPERPARAMS['batch_size'] + 1} salvato → {output_file}")

            LLMHandler.destroy_model(model, tokenizer)
            print("Model and tokenizer destroyed.")