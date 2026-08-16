import torch
import gc
from typing import Tuple, Optional, List, Dict, Any
from typing import Optional
import re
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import DECODER_SETTINGS
from src.model_backend import TextGenerationBackend

class DecoderHandler:
    PROMPT_STRATEGIES = [
        'no_perspective',
        'gender_male',
        'gender_female',
        'generation_boomer',
        'generation_gen_z',
        'generation_gen_y',
        'generation_gen_x',
        'relationship_colleagues',
        'relationship_friends',
    ]

    INPUT_TYPES = [
        'context',
        'context_reply',
        'conversation',
    ]

    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.training_params = DECODER_SETTINGS['training_params']
        self.inference_params = DECODER_SETTINGS['inference_params']
        self.output_dir = DECODER_SETTINGS['output_dir']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        load_dotenv()  
        login(token=os.getenv("HF_TOKEN"))

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def zero_shot(self, input_type: str, prompt_strategy: str):
        self._compute_inference(input_type, prompt_strategy)

    def _compute_inference(self, input_type: str, prompt_strategy: str):
        self.model, self.tokenizer, self.pipe = self._load_model()

        if self.model is not None and self.tokenizer is not None:
            print("\nModel and tokenizer loaded.")

        if input_type not in self.INPUT_TYPES:
            raise ValueError(f"Invalid input_type. Must be one of {self.INPUT_TYPES}")

        model_output_dir = self.output_dir / self.model_name.replace('/', '_')
        model_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = model_output_dir / f"{prompt_strategy}_{self.model_name.replace('/', '_')}_{self.dataset['name']}_{input_type}.csv"
        dataset_df, output_rows = self._get_existing_ids(output_file)

        for start in tqdm(range(0, len(dataset_df), self.inference_params['batch_size']), desc=f"Running {self.model_name}"):
            batch_df = dataset_df.iloc[start:start+self.inference_params['batch_size']]

            inputs = self._prepare_inputs(batch_df, input_type)
            results = self._generate_batch_output(inputs, prompt_strategy)

            cleaned_results = []
            for input, prompt, model_output, answer in results:
                invalid_inferences = 0
                invalid_model_outputs = [] 
                max_invalid_retries = self.inference_params.get(
                    'max_invalid_retries', 3
                )
                while (
                    answer not in ['0', '1']
                    and invalid_inferences < max_invalid_retries
                ):
                    invalid_inferences += 1
                    invalid_model_outputs.append(model_output)
                    _, prompt, model_output, answer = self._generate_batch_output([input], prompt_strategy)[0]

                if answer not in ['0', '1']:
                    invalid_model_outputs.append(model_output)
                    answer = None
                    print(
                        "\nWarning: no valid binary answer after "
                        f"{invalid_inferences + 1} attempts; "
                        "saving the inference as invalid.",
                        flush=True,
                    )

                cleaned_results.append((prompt, model_output, answer, invalid_inferences, invalid_model_outputs))

            for row, (prompt, model_output, answer, invalid_inferences, invalid_model_outputs) in zip(batch_df.itertuples(), cleaned_results):
                enriched = row._asdict()
                enriched['prompt'] = prompt
                enriched['model_output'] = model_output
                enriched['model_answer'] = answer
                enriched['invalid_inferences'] = invalid_inferences
                enriched['invalid_model_outputs'] = invalid_model_outputs
                output_rows.append(enriched)

            # Pulisci la memoria dopo ogni batch
            del results
            del cleaned_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # salva ad ogni batch
            output_df = pd.DataFrame(output_rows)
            
            # Salva CSV
            output_df.to_csv(
                output_file,
                index=False,
                encoding='utf-8'
            )
            
            print(f"\nBatch {start//self.inference_params['batch_size'] + 1} salvato → {output_file}")

        self._destroy_model()

    def fine_tuning(self):
        pass

    def _generate_batch_output(self, inputs, prompt_strategy: str) -> List[Dict[str, Any]]:
        """
        inputs: lista di conversazioni
        ritorna: lista di dict con prompt, model_output, answer
        """
        prompts = []
        for input in inputs:
            prompts.append(self._get_prompt(prompt_strategy, input))

        generated_texts = self.pipe.generate_texts(
            prompts,
            max_new_tokens=self.inference_params['max_new_tokens'],
            temperature=self.inference_params['temperature'],
        )

        results = []
        for input, prompt, model_output in zip(
            inputs, prompts, generated_texts
        ):

            answer = None
            answer = self._clean_output(f"<ANSWER>{model_output}", "answer")

            results.append((input, prompt, model_output, answer))

        # Pulisci la memoria dopo la generazione
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _get_prompt(self, prompt_strategy: str, input: str) -> str:
        if not prompt_strategy in self.PROMPT_STRATEGIES:
            raise ValueError(f"Not supported prompt strategy: {prompt_strategy}")
        
        if prompt_strategy == 'no_perspective':
            prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""

        elif prompt_strategy == 'gender_male':
            prompt = f"""<TASK_OVERVIEW>
You are male. Use this perspective when making the following decision. 
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
            
        elif prompt_strategy == 'gender_female':
            prompt = f"""<TASK_OVERVIEW>
You are female. Use this perspective when making the following decision. 
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
    
        elif prompt_strategy == 'generation_boomer':
            prompt = f"""<TASK_OVERVIEW>
You are Boomer. Use this perspective when making the following decision. 
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
            
        elif prompt_strategy == 'generation_gen_z':
            prompt = f"""<TASK_OVERVIEW>
You are Gen Z. Use this perspective when making the following decision.
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""

        elif prompt_strategy == 'generation_gen_y':
            prompt = f"""<TASK_OVERVIEW>
You are Gen Y. Use this perspective when making the following decision.
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""

        elif prompt_strategy == 'generation_gen_x':
            prompt = f"""<TASK_OVERVIEW>
You are Gen X. Use this perspective when making the following decision.
You have a pair of users, u1 and u2, involved in a dialogue. 
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""

        elif prompt_strategy == 'relationship_colleagues':
            prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. You know that u1 and u2 are colleagues.
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""

        elif prompt_strategy == 'relationship_friends':
            prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. You know that u1 and u2 are best friends.
The context contains two turns (where each turn encompasses a pair of messages between u1 and u2) and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
         
        return prompt

    def _clean_output(self, text: str, output_type: str) -> Optional[str]:
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

    def _get_existing_ids(self, output_file):
        output_rows = []

        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file, encoding='utf-8')
                existing_ids = set(existing_df['unique_id'].tolist())
                print(f"\nFound {len(existing_ids)} already processed entries in {output_file}")
                output_rows.extend(existing_df.to_dict('records')) # aggiungi i dati esistenti a output_rows
            except Exception as e:
                print(f"\nError loading existing CSV: {e}")
                existing_ids = set()
        else:
            existing_ids = set()

        dataset_df = pd.DataFrame(self.dataset['test'])
        print(f"\nTotal entries in test set: {len(dataset_df)}")
        dataset_df = dataset_df[~dataset_df['unique_id'].isin(existing_ids)] # filtra dataset_df per saltare già processati
        dataset_df = dataset_df.reset_index(drop=True)
        print(f"\nEntries to process: {len(dataset_df)}")

        return dataset_df, output_rows

    def _prepare_inputs(self, batch_df, input_type: str) -> List[str]:
            if input_type not in self.INPUT_TYPES:
                raise ValueError(f"Invalid input_type. Must be one of {self.INPUT_TYPES}")
            
            inputs = []

            if input_type == 'context':
                raise NotImplementedError("Input type 'context' is not implemented yet.")
                #if self.dataset['name'] == 'MultiPICo':
                #    inputs = batch_df['post'].tolist()
                #elif self.dataset['name'] == 'CSC':
                #    inputs = batch_df['context'].tolist()

            elif input_type == 'context_reply':
                raise NotImplementedError("Input type 'context_reply' is not implemented yet.")
                #if self.dataset['name'] == 'MultiPICo':
                #    inputs = (batch_df['post'] + "\nreply: " + batch_df['reply']).tolist()
                #elif self.dataset['name'] == 'CSC':
                #    inputs = (batch_df['context'] + "\nreply: " + batch_df['reply']).tolist()

            elif input_type == 'conversation':
                inputs = batch_df['dialogue'].apply(lambda d: d['turns']).tolist()

                cleaned_inputs = []
                for turns in inputs:
                    turns_copy = [dict(t) for t in turns]
                    # Rimuovi SOLO l'u2 dall'ULTIMO turno (la risposta target da predire)
                    if turns_copy and 'u2' in turns_copy[-1]:
                        del turns_copy[-1]['u2']

                    turns_str = ""
                    for turn in turns_copy:
                        # Ora alcuni turni avranno sia u1 che u2, l'ultimo solo u1
                        if 'u1' in turn:
                            turns_str += f"u1: {turn['u1']}\n"
                        if 'u2' in turn:
                            turns_str += f"u2: {turn['u2']}\n"
                    cleaned_inputs.append(turns_str)             

                inputs = cleaned_inputs

            return inputs

    def _load_model(self):
        print(f"\nCaricamento modello: {self.model_name}")
        backend = TextGenerationBackend(
            self.model_name, self.inference_params['dtype']
        )
        model, tokenizer, pipe = backend.load()

        print(f"\nModello caricato: {self.model_name}")
        print(f"\n  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        return model, tokenizer, pipe

    def _destroy_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None

        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nModello distrutto e cache GPU svuotata")
        else:
            print("\nModello distrutto")
