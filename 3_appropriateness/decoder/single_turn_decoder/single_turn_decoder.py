import json
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from tqdm import tqdm
from typing import Optional
from sklearn.model_selection import train_test_split

# check if gpu is available
device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

load_dotenv()  
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)

MODELS = [
    #'meta-llama/Llama-3.2-3B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
]

BATCH_SIZE = 8

def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # se manca pad token, usiamo eos come pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # impostiamo padding_side='left' per decoder-only mode
    tokenizer.padding_side = "left"

    # Creiamo la pipeline (la pipeline userà il tokenizer modificato)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=torch.float16 ,#"auto"
        #torch_dtype=torch.float16 #"auto"
    )

    # Assicuriamoci che la pipeline conosca il pad_token_id (alcune versioni lo richiedono esplicitamente)
    pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    return pipe, model, tokenizer



def destroy_model(model, tokenizer):
    if model is not None:
        del model
    
    if tokenizer is not None:
        del tokenizer

    torch.cuda.empty_cache()


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


def generate_batch_output(pipe, tokenizer, input_text_list, temperature: float = 0.5):
    """
    input_text_list: lista di frasi
    ritorna: lista di dict con prompt, model_output, reasoning, answer
    """
    prompts = []
    for text in input_text_list:
        prompts.append(f"""
        You have a pair of users, u1 and u2, who are having a conversation provided between the tags <CONVERSATION>...</CONVERSATION>. Determine whether it is appropriate for u2 to respond ironically to u1’s last message. 
        Output must be 0 (inappropriate) or 1 (appropriate).

        reason step by step, and then provide your final answer between the tags <ANSWER></ANSWER>.
        Provide your reasoning between the tags <REASONING></REASONING>.

        <CONVERSATION>u1: {text}</CONVERSATION>
        <REASONING>
        """)

    outputs = pipe(
        prompts,
        max_new_tokens=200,
        do_sample=temperature > 0,
        temperature=temperature,
        return_full_text=False,
        batch_size=BATCH_SIZE,
        pad_token_id=tokenizer.eos_token_id
    )

    results = []
    for text, prompt, out in zip(input_text_list, prompts, outputs):
        entry = out[0]
        model_output = entry['generated_text'].strip()
        reasoning = clean_output(f"<REASONING>{model_output}", "reasoning")
        answer = clean_output(model_output, "answer")
        results.append((text, prompt, model_output, reasoning, answer))

    return results


def calculate_aggregated_baseline(dataset_df: pd.DataFrame):
    for model_identifier in MODELS:
        print(f"Processing model: {model_identifier}")
        pipe, model, tokenizer = load_model(model_identifier)
        print("Model and tokenizer loaded.")

        output_file = f"./models/{model_identifier}_fine_tuned/predictions.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        output_rows = []

        # --- carica dati esistenti se presenti (CSV) ---
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file, dtype=str)
                existing_ids = set(existing_df['unique_id'])
                print(f"Found {len(existing_ids)} already processed entries in {output_file}")

                # aggiungi i dati esistenti a output_rows
                output_rows.extend(existing_df.to_dict(orient="records"))
            except Exception as e:
                print(f"Error loading existing CSV: {e}")
                existing_ids = set()
        else:
            existing_ids = set()

        # filtra dataset_df per saltare già processati
        dataset_df = dataset_df[~dataset_df['unique_id'].isin(existing_ids)]
        print(f"{len(dataset_df)} new entries to process")

        for start in tqdm(range(0, len(dataset_df), BATCH_SIZE), desc=f"Running {model_identifier}"):
            batch_df = dataset_df.iloc[start:start+BATCH_SIZE]
            posts = batch_df['text'].tolist()

            results = generate_batch_output(pipe, tokenizer, posts)

            cleaned_results = []
            for text, prompt, model_output, reasoning, answer in results:
                invalid_inferences = 0
                invalid_model_outputs = []
                while answer not in ['0', '1']:
                    invalid_inferences += 1
                    invalid_model_outputs.append(model_output)
                    text, prompt, model_output, reasoning, answer = generate_batch_output(pipe, tokenizer, [text])[0]
                cleaned_results.append((prompt, model_output, reasoning, answer, invalid_inferences, invalid_model_outputs))

            for row, (prompt, model_output, reasoning, answer, invalid_inferences) in zip(batch_df.itertuples(), cleaned_results):
                dict_row = row._asdict()
                enriched = {
                    'unique_id': dict_row['unique_id'],
                    'predicted_label': answer,
                    'label': dict_row['label'],
                    'prompt': prompt,
                    'model_output': model_output,
                    'model_reasoning': reasoning,
                    'invalid_inferences': invalid_inferences,
                    'invalid_model_outputs': invalid_model_outputs
                }
                output_rows.append(enriched)

            # salva ad ogni batch in CSV
            pd.DataFrame(output_rows).to_csv(
                output_file,
                index=False,
                encoding='utf-8'
            )

            print(f"✔ Batch {start//BATCH_SIZE + 1} salvato → {output_file}")

        destroy_model(model, tokenizer)
        print("Model and tokenizer destroyed.")


def load_dataset(file_path, language='en', dataset_name='MultiPICo', limit=None):
    """
    Carica il dataset dal unified corpus, mantenendo gli unique_id anche negli split.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            unified_corpus = json.load(f)
        print(f"Loaded unified corpus with {len(unified_corpus)} entries")
        
        df = pd.DataFrame(unified_corpus)
        
        # Filtra per lingua e dataset
        df_filtered = df[(df['language'] == language) & (df['dataset_name'] == dataset_name)]
        print(f"Filtered {dataset_name} {language.upper()} entries: {len(df_filtered)}")
        
        # Limita il numero di entries se necessario
        if limit is not None:
            df_filtered = df_filtered.head(limit)
            print(f"Using {len(df_filtered)} entries (limited)")

        # Seleziona solo le colonne utili
        df_filtered = df_filtered[['unique_id', 'post', 'annotations']].copy()
        df_filtered = df_filtered.rename(columns={'post': 'text'})
        df_filtered['label'] = df_filtered['annotations'].apply(
            lambda x: x.get('irony_appropriateness_label', None)
        )
        df_filtered = df_filtered.drop(columns=['annotations'])

        print(f"Entries with label 1: {len(df_filtered[df_filtered['label'] == 1])}")
        print(f"Entries with label 0: {len(df_filtered[df_filtered['label'] == 0])}")

        # Includiamo anche gli ID nello split
        train_df, temp_df = train_test_split(
            df_filtered,
            test_size=0.3,
            random_state=42,
            stratify=df_filtered['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )

        # Mostra distribuzioni
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f'{name} len: {len(split_df)} | label 1: {len(split_df[split_df["label"]==1])} | label 0: {len(split_df[split_df["label"]==0])}')

        dataset_splits = {
            'dataset_name': dataset_name,
            'dataset_language': language,
            'training_set': train_df.reset_index(drop=True),
            'validation_set': val_df.reset_index(drop=True),
            'test_set': test_df.reset_index(drop=True)
        }

        return dataset_splits
        
    except Exception as e:
        print(f"Error loading unified corpus: {e}")
        raise e

def main():
    data_path = "../../../1_datasets_merge/output/unified_corpus.json"

    # Carica e filtra il dataset
    dataset_splits = load_dataset(
        file_path=data_path,
        language='en',
        dataset_name='MultiPICo',
        limit=None  # None per usare tutti i dati, oppure un numero specifico per test
    )

    df_en_multipico = dataset_splits['test_set']

    calculate_aggregated_baseline(df_en_multipico)

if __name__ == "__main__":
    main()