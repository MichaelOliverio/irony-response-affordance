from typing import Optional
import pandas as pd
import re
import os
import json
import json5
from tqdm import tqdm

from src.config import DECODER_PARAMS, OUTPUT_DIR
from src.llm_handler import LLMHandler

class DecoderCoTClassifier:
    @staticmethod
    def clean_output(text: str, output_type: str) -> Optional[str]:
        """
        Estrae in modo robusto il contenuto tra <REASONING>...</REASONING> o <ANSWER>...</ANSWER>.
        Gestisce tag mancanti/troncati e differenze di maiuscole/minuscole.
        Restituisce None se non trova nulla.
        """
        if not text:
            return None

        flags = re.DOTALL | re.IGNORECASE

        if output_type == "reasoning":
            # 1) caso ideale: <REASONING>... </REASONING>
            m = re.search(r'<REASONING\s*>(.*?)</REASONING\s*>', text, flags)
            if m:
                return m.group(1).strip()

            # 2) se non c'è chiusura, estrai da <REASONING> fino al tag <ANSWER> (se presente)
            m = re.search(r'<REASONING\s*>(.*?)(?=<ANSWER\b)', text, flags)
            if m:
                return m.group(1).strip()

            # 3) se c'è solo <REASONING> e poi la fine del testo (troncato), prendi tutto il resto
            m = re.search(r'<REASONING\s*>(.*)$', text, flags)
            if m:
                return m.group(1).strip()

            return None

        elif output_type == "answer":
            # 1) caso ideale: <ANSWER>... </ANSWER>
            m = re.search(r'<ANSWER\s*>(.*?)</ANSWER\s*>', text, flags)
            if m:
                answer_text = m.group(1).strip()
                # Cerca solo 0 o 1 nella risposta
                digit_match = re.search(r'[01]', answer_text)
                if digit_match:
                    return digit_match.group(0)
                return answer_text

            # 2) se la chiusura manca, cerca una cifra 0/1 subito dopo <ANSWER>
            m = re.search(r'<ANSWER\s*>(\s*[01])\b', text, flags)
            if m:
                return m.group(1).strip()

            # 3) se c'è solo <ANSWER> e poi testo troncato -> prendi il resto e cerca 0/1 dentro
            m = re.search(r'<ANSWER\s*>(.*)$', text, flags)
            if m:
                candidate = m.group(1).strip()
                d = re.search(r'([01])', candidate)
                if d:
                    return d.group(1)
                return candidate  # fallback: ritorna quello che c'è

            # 4) fallback ulteriore: se non ci sono tag ma l'output finisce con una riga contenente solo 0 o 1
            m = re.search(r'\n\s*([01])\s*$', text, flags)
            if m:
                return m.group(1)

            return None

        else:
            raise ValueError("Invalid output_type. Must be 'answer' or 'reasoning'.")

    @staticmethod
    def generate_batch_output(pipe, tokenizer, input_conversation_list):
        """
        input_conversation_list: lista di conversazioni
        ritorna: lista di dict con prompt, model_output, reasoning, answer
        """
        prompts = []
        for conversation in input_conversation_list:
            prompts.append(f"""        
            You have a pair of users, u1 and u2, who are having a conversation provided between the tags <CONVERSATION>...</CONVERSATION>. Determine whether it is appropriate for u2 to respond ironically to u1’s last message. 
            Output must be 0 (inappropriate) or 1 (appropriate).

            reason step by step, and then provide your final answer between the tags <ANSWER></ANSWER>.
            Provide your reasoning between the tags <REASONING></REASONING>.

            <CONVERSATION>{conversation}</CONVERSATION>
            <REASONING>
            """)

        outputs = pipe(
            prompts,
            max_new_tokens=DECODER_PARAMS['max_new_tokens'],
            do_sample=DECODER_PARAMS['temperature'] > 0,
            temperature=DECODER_PARAMS['temperature'],
            return_full_text=False,
            batch_size=DECODER_PARAMS['batch_size'],
            pad_token_id=tokenizer.eos_token_id
        )

        results = []
        for prompt, out in zip(prompts, outputs):
            entry = out[0]
            model_output = entry['generated_text'].strip()
            reasoning = DecoderCoTClassifier.clean_output(f"<REASONING>{model_output}", "reasoning")
            answer = DecoderCoTClassifier.clean_output(model_output, "answer")
            results.append((prompt, model_output, reasoning, answer))

        return results
    
    @staticmethod
    def compute_predictions(dataset_df: pd.DataFrame, dataset: str, augmenter_model_name: str, added_turn: int):
        for model_name in DECODER_PARAMS['models']:
            print(f"Processing model: {model_name}")
            pipe, model, tokenizer = LLMHandler.load_model(model_name)
            print("Model and tokenizer loaded.")

            output_file = OUTPUT_DIR / f'cot_{model_name.replace("/", "_")}_{dataset}_num_gen_turn_{added_turn}_by_{augmenter_model_name}.json'
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
            # --- filtra record malformati senza dialogue.turns ---
            valid_mask = dataset_df['dialogue'].apply(
                lambda d: isinstance(d, dict) and 'turns' in d
            )

            num_invalid = (~valid_mask).sum()
            if num_invalid > 0:
                print(f"Scartati {num_invalid} record senza dialogue.turns")

            dataset_df = dataset_df[valid_mask]
            dataset_df = dataset_df.reset_index(drop=True)

            for start in tqdm(range(0, len(dataset_df), DECODER_PARAMS['batch_size']), desc=f"Running {model_name}"):
                batch_df = dataset_df.iloc[start:start+DECODER_PARAMS['batch_size']]

                conversations = batch_df['dialogue'].apply(lambda d: d['turns']).tolist()

                # Togliere "u2" dall’ultima turn di ogni dialogo
                cleaned_conversations = []
                for turns in conversations:
                    turns_copy = [dict(t) for t in turns]  # copia profonda
                    if turns_copy and 'u2' in turns_copy[-1]:
                        del turns_copy[-1]['u2']
                    cleaned_conversations.append(turns_copy)

                results = DecoderCoTClassifier.generate_batch_output(pipe, tokenizer, cleaned_conversations)

                cleaned_results = []
                for prompt, model_output, reasoning, answer in results:
                    invalid_inferences = 0
                    invalid_model_outputs = [] 
                    while answer not in ['0', '1']:
                        invalid_inferences += 1
                        invalid_model_outputs.append(model_output)
                        prompt, model_output, reasoning, answer = DecoderCoTClassifier.generate_batch_output(pipe, tokenizer, [prompt])[0]

                    cleaned_results.append((prompt, model_output, reasoning, answer, invalid_inferences, invalid_model_outputs))

                for row, (prompt, model_output, reasoning, answer, invalid_inferences, invalid_model_outputs) in zip(batch_df.itertuples(), cleaned_results):
                    enriched = row._asdict()
                    enriched['prompt'] = prompt
                    enriched['model_output'] = model_output
                    enriched['model_reasoning'] = reasoning
                    enriched['model_answer'] = answer
                    enriched['invalid_inferences'] = invalid_inferences
                    enriched['invalid_model_outputs'] = invalid_model_outputs
                    output_rows.append(enriched)

                # salva ad ogni batch
                pd.DataFrame(output_rows).to_json(
                    output_file,
                    orient='records',
                    indent=2,
                    force_ascii=False
                )
                print(f"Batch {start//DECODER_PARAMS['batch_size'] + 1} salvato → {output_file}")

            LLMHandler.destroy_model(model, tokenizer)
            print("Model and tokenizer destroyed.")

