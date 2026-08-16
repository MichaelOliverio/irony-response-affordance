import torch
from typing import Tuple, Optional, List, Dict, Any
from typing import Optional
from pathlib import Path
import re
import gc
from huggingface_hub import login
import shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from datasets import Dataset
import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import DECODER_SETTINGS
from src.model_backend import TextGenerationBackend

class DecoderHandler:
    PROMPT_STRATEGIES = [
        'zero_shot',
        'chain_of_thought',
    ]

    INPUT_TYPES = [
        'last_u1_message',
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

        # Configurazione gestione memoria PyTorch
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        #if torch.cuda.is_available():
        #    torch.set_default_dtype(torch.bfloat16)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def zero_shot(self, input_type: str):
        """Inferenza zero-shot con modello base (non fine-tunato)."""
        self._inference(input_type, prompt_strategy='zero_shot', is_fine_tuned=False)

    def chain_of_thought(self, input_type: str):
        """Inferenza chain-of-thought con modello base (non fine-tunato)."""
        self._inference(input_type, prompt_strategy='chain_of_thought', is_fine_tuned=False)

    def fine_tuned_inference(self, input_type: str, fine_tuned_model_path: str = None):
        """
        Inferenza zero-shot con modello fine-tunato.
        
        Args:
            input_type: tipo di input ('last_u1_message', 'context_reply', 'conversation')
            fine_tuned_model_path: percorso del modello fine-tunato (opzionale, auto-detected)
        """
        self._inference(
            input_type, 
            prompt_strategy='zero_shot', 
            is_fine_tuned=True, 
            fine_tuned_model_path=fine_tuned_model_path
        )

    def _inference(self, input_type: str, prompt_strategy: str, is_fine_tuned: bool = False, fine_tuned_model_path: str = None):
        """
        Metodo unificato per l'inferenza.
        
        Args:
            input_type: tipo di input ('last_u1_message', 'context_reply', 'conversation')
            prompt_strategy: strategia di prompt ('zero_shot', 'chain_of_thought')
            is_fine_tuned: se True, usa un modello fine-tunato
            fine_tuned_model_path: percorso del modello fine-tunato (per is_fine_tuned=True)
        """
        # Caricamento modello
        if is_fine_tuned:
            # Determina il path del modello fine-tunato
            if fine_tuned_model_path is None:
                clean_model_name = self.model_name.replace('/', '_')
                dataset_name = self.dataset['name']
                fine_tuned_model_path = str(
                    self.training_params['training_args']['output_dir'] / clean_model_name / dataset_name / "final_model"
                )
            
            if not Path(fine_tuned_model_path).exists():
                raise FileNotFoundError(
                    f"Fine-tuned model not found at: {fine_tuned_model_path}\n"
                    f"Please run fine_tuning() first or provide a valid model path."
                )
            
            print(f"\n{'='*60}")
            print(f"Starting FINE-TUNED inference")
            print(f"Model: {self.model_name}")
            print(f"Fine-tuned model path: {fine_tuned_model_path}")
            print(f"Prompt strategy: {prompt_strategy}")
            print(f"Input type: {input_type}")
            print(f"{'='*60}\n")
            
            self.model, self.tokenizer = self._load_fine_tuned_model(fine_tuned_model_path)
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=self.inference_params['torch_dtype'],
                pad_token_id=self.tokenizer.pad_token_id,
            )
            print("Fine-tuned model and pipeline loaded successfully.\n")
        else:
            print(f"\n{'='*60}")
            print(f"Starting BASELINE inference")
            print(f"Model: {self.model_name}")
            print(f"Prompt strategy: {prompt_strategy}")
            print(f"Input type: {input_type}")
            print(f"{'='*60}\n")
            
            self.model, self.tokenizer, self.pipe = self._load_model()

        if self.model is not None and self.tokenizer is not None:
            print("Model and tokenizer loaded successfully.\n")

        # Validazione input_type
        if input_type not in self.INPUT_TYPES:
            raise ValueError(f"Invalid input_type. Must be one of {self.INPUT_TYPES}")

        clean_model_name = self.model_name.replace('/', '_')
        model_output_dir = self.output_dir / clean_model_name
        if is_fine_tuned:
            model_output_dir = model_output_dir / "fine_tuned"
        model_output_dir.mkdir(parents=True, exist_ok=True)

        evaluation_sets = [('dataset_ground_truth', 'test', '')]
        if not is_fine_tuned and self.dataset.get('human_test'):
            evaluation_sets.append(
                ('human_majority', 'human_test', '_human_majority')
            )

        for reference_name, split_name, filename_suffix in evaluation_sets:
            if is_fine_tuned:
                filename = (
                    f"{prompt_strategy}_{clean_model_name}_{self.dataset['name']}_"
                    f"{input_type}_finetuned.csv"
                )
            else:
                filename = (
                    f"{prompt_strategy}_{clean_model_name}_{self.dataset['name']}_"
                    f"{input_type}{filename_suffix}.csv"
                )
            output_file = model_output_dir / filename
            dataset_df, output_rows = self._get_existing_ids(
                output_file, split_name
            )

            for start in tqdm(
                range(0, len(dataset_df), self.inference_params['batch_size']),
                desc=(
                    f"Running {'fine-tuned' if is_fine_tuned else 'baseline'} "
                    f"{self.model_name} [{reference_name}]"
                ),
            ):
                batch_df = dataset_df.iloc[
                    start:start + self.inference_params['batch_size']
                ]
                inputs = self._prepare_inputs(batch_df, input_type)
                results = self._generate_batch_output(
                    inputs, input_type, prompt_strategy
                )

                cleaned_results = []
                for input_text, prompt, model_output, reasoning, answer in results:
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
                        (
                            _,
                            prompt,
                            model_output,
                            reasoning,
                            answer,
                        ) = self._generate_batch_output(
                            [input_text], input_type, prompt_strategy
                        )[0]
                    if answer not in ['0', '1']:
                        invalid_model_outputs.append(model_output)
                        answer = None
                        print(
                            "\nWarning: no valid binary answer after "
                            f"{invalid_inferences + 1} attempts; "
                            "saving the inference as invalid.",
                            flush=True,
                        )
                    cleaned_results.append(
                        (
                            prompt,
                            model_output,
                            reasoning,
                            answer,
                            invalid_inferences,
                            invalid_model_outputs,
                        )
                    )

                for row, generated in zip(
                    batch_df.itertuples(), cleaned_results
                ):
                    (
                        prompt,
                        model_output,
                        reasoning,
                        answer,
                        invalid_inferences,
                        invalid_model_outputs,
                    ) = generated
                    enriched = row._asdict()
                    enriched['evaluation_reference'] = reference_name
                    enriched['prompt'] = prompt
                    enriched['model_output'] = model_output
                    enriched['model_reasoning'] = reasoning
                    enriched['model_answer'] = answer
                    enriched['invalid_inferences'] = invalid_inferences
                    enriched['invalid_model_outputs'] = invalid_model_outputs
                    output_rows.append(enriched)

                pd.DataFrame(output_rows).to_csv(
                    output_file, index=False, encoding='utf-8'
                )
                print(
                    f"\nBatch "
                    f"{start // self.inference_params['batch_size'] + 1} "
                    f"saved -> {output_file}"
                )
                if (
                    start // self.inference_params['batch_size'] + 1
                ) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print(f"\nInference completed [{reference_name}]")
            print(f"Output saved to: {output_file}")

        self._destroy_model()

    def fine_tuning(self, input_type: str, force_retrain: bool = False, resume_from_checkpoint: bool = True):
        """
        Esegue il fine-tuning del modello usando LoRA con strategia zero-shot.
        
        Args:
            input_type: tipo di input usato per il training ('last_u1_message' o 'conversation')
            force_retrain: se True, elimina il modello esistente e riallena da zero
            resume_from_checkpoint: se True, riprende dall'ultimo checkpoint se disponibile
        
        Returns:
            str: percorso del modello fine-tunato
        """
        from peft import LoraConfig
        from transformers import TrainingArguments
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

        if input_type not in ['last_u1_message', 'conversation']:
            raise ValueError(f"Invalid input_type for fine-tuning. Must be 'last_u1_message' or 'conversation', got: {input_type}")
        
        print(f"\n{'='*60}")
        print(f"Starting fine-tuning for {self.model_name}")
        print(f"Dataset: {self.dataset['name']}")
        print(f"Input type: {input_type}")
        print(f"{'='*60}\n")

        # Configurazione path - include input_type nel path
        clean_model_name = self.model_name.replace('/', '_')
        dataset_name = self.dataset['name']
        output_path = self.training_params['training_args']['output_dir'] / clean_model_name / dataset_name / input_type
        final_model_path = output_path / "final_model"
        
        # Verifica se il modello esiste già
        if final_model_path.exists() and not force_retrain:
            print(f"\n     Fine-tuned model already exists: {final_model_path}")
            print("Skipping fine-tuning. Use force_retrain=True to retrain from scratch.")
            return str(final_model_path)
        
        # Se force_retrain, elimina il modello esistente
        if force_retrain and output_path.exists():
            print(f"\n     Removing existing model: {output_path}")
            shutil.rmtree(output_path)
            print("✓ Old model removed")
        
        output_path.mkdir(parents=True, exist_ok=True)

        # Preparazione dataset
        train_dataset = self._prepare_training_dataset(self.dataset['train'], input_type)
        eval_dataset = self._prepare_training_dataset(self.dataset['val'], input_type)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        # Salva input_type per usarlo in _formatting_prompts_func
        self._current_input_type = input_type
        
        # Caricamento modello base
        self.model, self.tokenizer = self._load_model_for_training()
        
        # Configurazione LoRA
        peft_config = LoraConfig(**self.training_params['lora'])
        
        # Configurazione training arguments
        training_args_dict = self.training_params['training_args'].copy()
        training_args_dict['output_dir'] = str(output_path)
        training_args_dict['logging_dir'] = str(output_path / "logs")
        training_arguments = TrainingArguments(**training_args_dict)
        
        # Data collator per completion-only training
        response_template = "<ANSWER>"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        
        # Verifica se ci sono checkpoint da cui riprendere
        checkpoint_dir = None
        if resume_from_checkpoint and output_path.exists():
            checkpoints = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
            if checkpoints:
                # Ordina per numero di step e prendi l'ultimo
                checkpoints.sort(key=lambda x: int(x.name.split('-')[-1]))
                checkpoint_dir = str(checkpoints[-1])
                print(f"\n     Resuming from checkpoint: {checkpoint_dir}")
        
        # Trainer con max_seq_length
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            args=training_arguments,
            formatting_func=self._formatting_prompts_func,
            data_collator=collator,
            tokenizer=self.tokenizer,  # Esplicitamente specificato per evitare warning
        )
        
        # Training
        print("\nStarting training...")
        if checkpoint_dir:
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        else:
            trainer.train()
        
        # Salvataggio modello finale
        trainer.save_model(str(final_model_path))
        print(f"\n✓ Model saved to: {final_model_path}")
        
        # Salva anche le metriche di training
        metrics_file = output_path / "training_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(trainer.state.log_history, f, indent=2, default=str)
        print(f"✓ Training metrics saved to: {metrics_file}")
        
        # Cleanup
        del trainer
        self._destroy_model()
        
        print(f"\n{'='*60}")
        print(f"Fine-tuning completed for {self.model_name}")
        print(f"{'='*60}\n")
        
        return str(final_model_path)

    def _generate_batch_output(self, inputs, input_type:str, prompt_strategy: str) -> List[Dict[str, Any]]:
        """
        inputs: lista di conversazioni
        ritorna: lista di dict con prompt, model_output, reasoning, answer
        """
        prompts = []
        for input in inputs:
            prompts.append(self._get_prompt(prompt_strategy, input_type, input))

        if isinstance(self.pipe, TextGenerationBackend):
            generated_texts = self.pipe.generate_texts(
                prompts,
                max_new_tokens=self.inference_params['max_new_tokens'],
                temperature=self.inference_params['temperature'],
            )
        else:
            outputs = self.pipe(
                prompts,
                max_new_tokens=self.inference_params['max_new_tokens'],
                do_sample=self.inference_params['temperature'] > 0,
                temperature=self.inference_params['temperature'],
                return_full_text=False,
                batch_size=self.inference_params['batch_size'],
                pad_token_id=self.tokenizer.pad_token_id
            )
            generated_texts = [
                output[0]['generated_text'].strip() for output in outputs
            ]

        results = []
        for input, prompt, model_output in zip(
            inputs, prompts, generated_texts
        ):

            answer, reasoning = None, None
            if prompt_strategy in ['zero_shot']:
                answer = self._clean_output(f"<ANSWER>{model_output}", "answer")
            elif prompt_strategy in ['chain_of_thought']:
                answer = self._clean_output(model_output, "answer")
                reasoning = self._clean_output(f"<REASONING>{model_output}", "reasoning")

            results.append((input, prompt, model_output, reasoning, answer))

        return results

    def _get_prompt(self, prompt_strategy: str, input_type: str, input: str) -> str:
        if not prompt_strategy in self.PROMPT_STRATEGIES:
            raise ValueError(f"Not supported prompt strategy: {prompt_strategy}")
        if not input_type in self.INPUT_TYPES:
            raise ValueError(f"Not supported input type: {input_type}")
        
        if self.dataset['name'] == 'CSC':
            prompt = self._get_csc_prompt(prompt_strategy, input_type, input)
        else:
            prompt = self._get_multipico_prompt(prompt_strategy, input_type, input)

        return prompt

    def _get_multipico_prompt(self, prompt_strategy: str, input_type: str, input: str) -> str:
        prompt = ""
        if input_type == 'context_reply': # upper bound
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a pair of messages exchanged on social media.
Taking into account u2's observed reply, would an ironic response by u2 have been a plausible and natural reply to u1's preceding message?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a pair of messages exchanged on social media.
Taking into account u2's observed reply, would an ironic response by u2 have been a plausible and natural reply to u1's preceding message?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The reasoning is:
<REASONING>"""
                
        elif input_type == 'last_u1_message': # baseline
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a message from u1 on social media, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a message from u1 on social media, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The reasoning is:
<REASONING>"""

        elif input_type == 'conversation': # full context
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of two turns and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of two turns and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The reasoning is:
<REASONING>"""
            
        return prompt

    def _get_csc_prompt(self, prompt_strategy: str, input_type: str, input: str) -> str:
        prompt = ""
        if input_type == 'context_reply': # upper bound
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a description of the situation and the observed reply of u2.
Taking into account u2's observed reply, would an ironic response by u2 have been a plausible and natural reply in the preceding situation?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a description of the situation and the observed reply of u2.
Taking into account u2's observed reply, would an ironic response by u2 have been a plausible and natural reply in the preceding situation?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The reasoning is:
<REASONING>"""
                
        elif input_type == 'last_u1_message': # baseline
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of a message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The reasoning is:
<REASONING>"""

        elif input_type == 'conversation': # full context
            if prompt_strategy == 'zero_shot':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of two turns and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.

<CONTEXT>{input}</CONTEXT>
</TASK_OVERVIEW>

The answer is:
<ANSWER>"""
                
            elif prompt_strategy == 'chain_of_thought':
                prompt = f"""<TASK_OVERVIEW>
You have a pair of users, u1 and u2, involved in a dialogue. 
The context, shown between the tags <CONTEXT>...</CONTEXT>, consists of two turns and a last message from u1, to which u2 has to respond.
Given this dialogue history, would an ironic response be a plausible and natural continuation for u2, the upcoming respondent?
The output must be 0 (not afforded) or 1 (afforded).
Provide your output answer between the tags <ANSWER></ANSWER>.
Provide your reasoning between the tags <REASONING></REASONING>.

<CONTEXT>{input}</CONTEXT>
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

    def _get_existing_ids(self, output_file, split_name='test'):
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

        dataset_df = pd.DataFrame(self.dataset[split_name])
        print(f"\nTotal entries in {split_name}: {len(dataset_df)}")
        dataset_df = dataset_df[~dataset_df['unique_id'].isin(existing_ids)] # filtra dataset_df per saltare già processati
        dataset_df = dataset_df.reset_index(drop=True)
        print(f"\nEntries to process: {len(dataset_df)}")

        return dataset_df, output_rows

    def _prepare_inputs(self, batch_df, input_type: str) -> List[str]:
            if input_type not in self.INPUT_TYPES:
                raise ValueError(f"Invalid input_type. Must be one of {self.INPUT_TYPES}")
            
            inputs = []

            if input_type == 'last_u1_message':
                if self.dataset['name'] == 'MultiPICo':
                    inputs = ("\nu1: " + batch_df['post']).tolist()
                elif self.dataset['name'] == 'CSC':
                    inputs = batch_df['dialogue'].apply(lambda d: d['turns']).tolist()
                    cleaned_inputs = []
                    for turns in inputs:
                        turns_str = "u1:" + turns[-1]['u1']  # Prendi solo l'ultimo messaggio di u1
                        cleaned_inputs.append(turns_str)
                    inputs = cleaned_inputs

            elif input_type == 'context_reply':
                if self.dataset['name'] == 'MultiPICo':
                    inputs = ("\nu1: " + batch_df['post'] + "\nu2: " + batch_df['reply']).tolist()
                elif self.dataset['name'] == 'CSC':
                    inputs = ("\ndescription of the situation: " + batch_df['context'] + "\nu2: " + batch_df['reply']).tolist()

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
            self.model_name, self.inference_params['torch_dtype']
        )
        model, tokenizer, pipe = backend.load()

        print(f"\nModello caricato: {self.model_name}")
        print(f"\n  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        return model, tokenizer, pipe

    def _load_model_for_training(self):
        """Carica il modello base per il fine-tuning."""
        print(f"\nLoading model for training: {self.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Configurazione padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        
        print(f"Model loaded for training: {self.model_name}")
        return model, tokenizer

    def _load_fine_tuned_model(self, fine_tuned_model_path: str):
        """
        Carica un modello fine-tunato con LoRA, fa il merge degli adapter e lo prepara per l'inferenza.
        
        Args:
            fine_tuned_model_path: percorso del modello fine-tunato
        
        Returns:
            model, tokenizer
        """
        from peft import PeftModel

        print(f"\nLoading fine-tuned model from: {fine_tuned_model_path}")
        
        # Carica il modello base
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.inference_params['torch_dtype'],
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        
        # Carica gli adapter LoRA
        model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
        
        # Merge degli adapter con il modello base
        model = model.merge_and_unload()
        
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # Configurazione padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        
        print(f"Fine-tuned model loaded and merged successfully")
        return model, tokenizer

    def _prepare_training_dataset(self, dataset_split, input_type: str):
        """
        Prepara il dataset per il training in formato Dataset di Hugging Face con strategia zero-shot.
        
        Args:
            dataset_split: split del dataset (train/val/test)
            input_type: tipo di input ('last_u1_message' o 'conversation')
        
        Returns:
            Dataset Hugging Face
        """
        df = pd.DataFrame(dataset_split)
        
        # Usa _prepare_inputs per ottenere gli input formattati
        inputs = self._prepare_inputs(df, input_type)
        
        # Crea il dataset per il training
        conversations = []
        for input_text, (_, row) in zip(inputs, df.iterrows()):
            conversations.append({
                'conversation': input_text,
                'label': str(row['label'])
            })
        
        return Dataset.from_pandas(pd.DataFrame(conversations))

    def _formatting_prompts_func(self, examples):
        """
        Formatta i prompt per il training con strategia zero-shot.
        Riutilizza _get_prompt per garantire coerenza con l'inferenza.
        
        Args:
            examples: batch di esempi
        
        Returns:
            lista di prompt formattati
        """
        formatted_prompts = []
        
        for conv, label in zip(examples['conversation'], examples['label']):
            # Usa _get_prompt per ottenere il prompt base (già include <ANSWER> senza chiusura)
            base_prompt = self._get_prompt('zero_shot', self._current_input_type, conv)
            
            # Aggiungi la label alla fine (base_prompt finisce già con "<ANSWER>")
            prompt = f"{base_prompt}{label}</ANSWER>"
            
            formatted_prompts.append(prompt)
        
        return formatted_prompts

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
