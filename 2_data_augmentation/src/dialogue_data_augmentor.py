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
            if dataset_name == "MultiPICo":
                initial_output = """<JSON>{"turns": [{"u1":\""""
                dialogue = JSONParser.multipico_parse(f"{initial_output}{model_output}", num_turns, posts[i], replies[i])
            elif dataset_name == "CSC":
                initial_output = """<JSON>{"turns": [{"Steve":\""""
                dialogue = JSONParser.csc_parse(f"{initial_output}{model_output}", num_turns, replies[i])
            
            results.append((prompt, model_output, dialogue))
            i += 1
        return results
    
    @staticmethod
    def multipico_prompt(post, reply, num_turns) -> str:
        initial_prompt = f"""
        You are given the following pair of final messages exchanged on social media between user u1 and user u2:

        u1: {post}
        u2: {reply}

        Your task is to generate EXACTLY {num_turns} dialogue turns (each turn = one message from u1 and one message from u2) that logically and stylistically lead up to the given final messages.

        STRICT REQUIREMENTS:
        - You MUST output exactly {num_turns + 1} turns total: the {num_turns} generated turns + the final provided turn.
        - All generated messages MUST be coherent, consistent, and plausible precursors to the final turn.
        - Absolutely DO NOT invent or repeat text from the final messages.
        - DO NOT repeat messages across different turns.
        - DO NOT output placeholder or generic text such as \"First message from u1\".
        - DO NOT produce duplicated messages or self-repetitions across any turn.
        - Preserve the writing style, tone, sentiment, and personality of u1 and u2.
        - Make the conversation flow naturally towards the final turn.
        
        OUTPUT FORMAT (MANDATORY):
        Return ONLY a JSON object in this exact structure:
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

        Produce only valid JSON. No explanations, no comments, no extra text.
        RESPONSE:
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
        Given the context and the reply of user u2 to Steve (whose relation to u2 is: {relation}):

        context: {context}
        u2: {reply}

        Your task is to generate EXACTLY {num_turns} preceding dialogue turns (each turn consisting of a message from Steve followed by a message from u2), PLUS the Steve message that directly precedes the final provided u2 reply.
        
        STRICT REQUIREMENTS:
        - You MUST output exactly {num_turns + 1} turns total:
        - {num_turns} generated turns
        - + 1 final turn consisting of:
                Steve: the message that directly precedes the final given u2 line
                u2: the provided reply ({reply})
        - All generated messages MUST be coherent, stylistically consistent, and logically lead to the provided reply.
        - Absolutely DO NOT invent or repeat any part of the provided reply.
        - DO NOT output placeholders or generic fillers (e.g., \"First message from u1\").
        - DO NOT repeat messages across different turns.
        - DO NOT produce duplicated messages or self-repetitions across any turn.
        - Preserve the writing style, tone, sentiment, and personality of u2.
        - Messages from u1 must sound like Steve and must fit the relation: {relation}.
        - The context must be respected and smoothly woven into the preceding turns.
        - Output MUST be valid JSON ONLY. No comments, no extra explanations.

        MANDATORY OUTPUT FORMAT:
        <JSON>
        {{
            "turns": ["""

        # turni
        turni_prompt = ""
        for i in range(num_turns):
            turni_prompt += f"""
            {{
                "Steve": "{PROMPT_MAPPING[i+1]} message from Steve",
                "u2": "{PROMPT_MAPPING[i+1]} message from u2"
            }},"""
            
        # Ultimo turno
        turni_prompt += f"""
        {{
            "Steve": "Message from Steve that precedes the given u2",
            "u2": "{reply}"
        }}"""

        final_prompt = f"""
            ]
        }}
        </JSON>

        Produce only valid JSON. No explanations, no comments, no extra text.
        RESPONSE:
        <JSON>
        {{
            "turns": [
                {{
                    "Steve": "
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
                print(f"Generating dialogues with {i + 1} turns...\n")

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
                skew_dataset_df = dataset_df[~dataset_df['unique_id'].isin(existing_ids)]
                print(f"{len(skew_dataset_df)} new entries to process")

                for start in tqdm(range(0, len(skew_dataset_df), DATA_AUGMENTATION_HYPERPARAMS['batch_size']), desc=f"Running {model_name}"):
                    batch_df = skew_dataset_df.iloc[start:start+DATA_AUGMENTATION_HYPERPARAMS['batch_size']].copy()
                    
                    # Rimuovi @USER dai dati e aggiorna il DataFrame
                    if dataset_name == "MultiPICo":
                        batch_df['post'] = batch_df['post'].apply(lambda p: re.sub(r'@USER\s*', '', p))
                        post = batch_df['post'].tolist()
                    elif dataset_name == "CSC":
                        post = batch_df['context'].tolist()

                    batch_df['reply'] = batch_df['reply'].apply(lambda r: re.sub(r'@USER\s*', '', r))
                    reply = batch_df['reply'].tolist()

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