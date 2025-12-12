import torch
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

class DecoderHandler:
    PROMPT_STRATEGIES = [
        'zero_shot',
        'chain_of_thought',
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

    def zero_shot(self, input_type: str):
        self.model, self.tokenizer, self.pipe = self._load_model()

        if self.model is not None and self.tokenizer is not None:
            print("\nModel and tokenizer loaded.")

        if input_type not in self.INPUT_TYPES:
            raise ValueError(f"Invalid input_type. Must be one of {self.INPUT_TYPES}")

        output_file = self.output_dir / self.model_name.replace('/', '_') / f"zero_shot_{self.model_name.replace('/', '_')}_{self.dataset}_{input_type}.json"
        dataset_df, output_rows = self._get_existing_ids(output_file)

        for start in tqdm(range(0, len(dataset_df), self.inference_params['batch_size']), desc=f"Running {self.model_name}"):
            batch_df = dataset_df.iloc[start:start+self.inference_params['batch_size']]

            inputs = self._prepare_inputs(batch_df, input_type)
            results = self._generate_batch_output(inputs, 'zero_shot')

            cleaned_results = []
            for input, prompt, model_output, reasoning, answer in results:
                invalid_inferences = 0
                invalid_model_outputs = [] 
                while answer not in ['0', '1']:
                    invalid_inferences += 1
                    invalid_model_outputs.append(model_output)
                    _, prompt, model_output, reasoning, answer = self._generate_batch_output([input], 'zero_shot')[0]

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
            print(f"\nBatch {start//self.inference_params['batch_size'] + 1} salvato → {output_file}")

        self._destroy_model()

    def fine_tuning(self):
        pass

    def _generate_batch_output(self, inputs, prompt_strategy: str) -> List[Dict[str, Any]]:
        """
        inputs: lista di conversazioni
        ritorna: lista di dict con prompt, model_output, reasoning, answer
        """
        prompts = []
        for input in inputs:
            prompts.append(self._get_prompt(prompt_strategy, input))

        outputs = self.pipe(
            prompts,
            max_new_tokens=self.inference_params['max_new_tokens'],
            do_sample=self.inference_params['temperature'] > 0,
            temperature=self.inference_params['temperature'],
            return_full_text=False,
            batch_size=self.inference_params['batch_size'],
            pad_token_id=self.tokenizer.pad_token_id
        )

        results = []
        for prompt, out in zip(prompts, outputs):
            entry = out[0]
            model_output = entry['generated_text'].strip()

            answer, reasoning = None, None
            if prompt_strategy in ['zero_shot']:
                answer = self._clean_output(f"<ANSWER>{model_output}", "answer")
            elif prompt_strategy in ['chain_of_thought']:
                answer = self._clean_output(model_output, "answer")
                reasoning = self._clean_output(f"<REASONING>{model_output}", "reasoning")

            results.append((input, prompt, model_output, reasoning, answer))

        return results

    def _get_prompt(self, prompt_strategy: str, input: str) -> str:
        if not prompt_strategy in self.PROMPT_STRATEGIES:
            raise ValueError(f"Not supported prompt strategy: {prompt_strategy}")
        
        if prompt_strategy == 'zero_shot':
            prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, who are having a conversation provided between the tags <CONVERSATION>...</CONVERSATION>.
Determine whether it is appropriate for u2 to respond ironically to u1’s last message.
Output must be 0 (inappropriate) or 1 (appropriate).
Provide your answer between the tags <ANSWER></ANSWER>.
<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>
The answer is:
<ANSWER>"""
            
        elif prompt_strategy == 'chain_of_thought':
            prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, who are having a conversation provided between the tags <CONVERSATION>...</CONVERSATION>.
Determine whether it is appropriate for u2 to respond ironically to u1’s last message.
Output must be 0 (inappropriate) or 1 (appropriate).
Reason step by step, and then provide your final answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.
<CONVERSATION>{input}</CONVERSATION>
</TASK_OVERVIEW>
The reasoning is:
<REASONING>"""
            
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
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_ids = set([row['unique_id'] for row in existing_data])
                print(f"\nFound {len(existing_ids)} already processed entries in {output_file}")
                output_rows.extend(existing_data) # aggiungi i dati esistenti a output_rows
            except Exception as e:
                print(f"\nError loading existing JSON: {e}")
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
                if self.dataset['name'] == 'MultiPICo':
                    inputs = batch_df['post'].tolist()
                elif self.dataset['name'] == 'CSC':
                    inputs = batch_df['context'].tolist()

            elif input_type == 'context_reply':
                if self.dataset['name'] == 'MultiPICo':
                    inputs = (f"post: {batch_df['post']}\nreply{batch_df['reply']}").tolist()
                elif self.dataset['name'] == 'CSC':
                    inputs = (f"context: {batch_df['context']}\nreply{batch_df['reply']}").tolist()

            elif input_type == 'conversation':
                inputs = batch_df['dialogue'].apply(lambda d: d['turns']).tolist()

                cleaned_inputs = []
                for turns in inputs:
                    turns_copy = [dict(t) for t in turns]  # copia profonda
                    if turns_copy and 'u2' in turns_copy[-1]:
                        del turns_copy[-1]['u2']

                    turns_str = ""
                    for turn in turns_copy:
                        speaker = 'u1' if 'u1' in turn else 'u2'
                        content = turn[speaker]
                        turns_str += f"{speaker}: {content}\n"
                    cleaned_inputs.append(turns_str)            

                inputs = cleaned_inputs

            return inputs

    def _load_model(self):
        print(f"\nCaricamento modello: {self.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Se manca pad token, usiamo eos come pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Impostiamo padding_side='left' per decoder-only models
        tokenizer.padding_side = "left"

        # Creiamo la pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=self.inference_params['dtype'],
            pad_token_id=tokenizer.pad_token_id,
        )

        print(f"\nModello caricato: {self.model_name}")
        print(f"\n  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        return model, tokenizer, pipe

    def _destroy_model(self):
        if self.model is not None:
            del self.model
        
        if self.tokenizer is not None:
            del self.tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\nModello distrutto e cache GPU svuotata")
        else:
            print("\nModello distrutto")