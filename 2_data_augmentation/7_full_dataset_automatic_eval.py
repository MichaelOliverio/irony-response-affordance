"""
Automatic evaluation of generated dialogues using perplexity and LLM-as-a-judge.
"""
import json
import re
from pathlib import Path
import pandas as pd
from src import DialogueEvaluator
from src.config import INVALID_UNIQUEID

def parse_filename(filename):
    """Parse filename to extract dataset name, model name and number of turns"""
    pattern = re.compile(
        r'^(?P<dataset>[^_]+)_(?P<model>.+?)_n_turn_(?P<turn>\d+).json$'
    )
    
    match = pattern.match(filename)
    if match:
        return match.group("dataset"), match.group("model"), int(match.group("turn"))
    else:
        return None

def remove_invalid_entries(df):
    """Remove entries with invalid unique IDs."""
    initial_count = len(df)
    df_cleaned = df[~df['unique_id'].isin(INVALID_UNIQUEID)].reset_index(drop=True)
    removed_count = initial_count - len(df_cleaned)
    print(f"Removed {removed_count} invalid entries based on unique_id")
    return df_cleaned

def remove_short_context_entries(df, dataset_name):
    # voglio rimuovere quei dialoghi di multipico che hanno una sola parola come post o come reply oppure che hanno solamente punteggiatura

    def is_too_short(text):
        if not isinstance(text, str):
            return True
        text = text.strip()
        if len(text.split()) <= 1:
            return True
        if all(char in '.,!?;:' for char in text):
            return True
        return False
    
    
    initial_count = len(df)
    if dataset_name == 'MultiPICo':
        mask = df.apply(lambda row: is_too_short(row['post']) or is_too_short(row['reply']), axis=1)
    else:
        mask = df.apply(lambda row: is_too_short(row['reply']), axis=1)

    # Get the unique_ids of removed entries
    removed_unique_ids = df[mask]['unique_id'].tolist()
    
    df_cleaned = df[~mask].reset_index(drop=True)
    removed_count = initial_count - len(df_cleaned)
    print(f"Removed {removed_count} entries with too short context or reply")
    return df_cleaned, removed_unique_ids

def main():
    """Main function for automatic evaluation."""
    output_dir = Path("output")
    eval_output_dir = output_dir / "evaluations"
    eval_output_dir.mkdir(exist_ok=True)
    
    # Create directory for removed entries logs
    removed_entries_dir = eval_output_dir / "removed_entries"
    removed_entries_dir.mkdir(exist_ok=True)
    
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in output directory")
        return
    
    print(f"{'='*70}")
    print(f"Automatic Dialogue Evaluation")
    print(f"{'='*70}")
    print(f"Found {len(json_files)} augmented dataset(s) to evaluate\n")
    
    # Initialize evaluator
    evaluator = DialogueEvaluator(
        judge_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
        ]
    )
    
    for json_file in json_files:
        print(f"\n{'='*70}")
        print(f"Processing: {json_file.name}")
        print(f"{'='*70}")
        
        # Parse filename
        parsed = parse_filename(json_file.name)
        if not parsed:
            print(f"Skipping {json_file.name} - doesn't match expected pattern")
            continue
        
        dataset_name, model_name, added_turn = parsed
        print(f"Dataset: {dataset_name}")
        print(f"Augmenter Model: {model_name}")
        print(f"Added Turns: {added_turn}")
        
        # Load data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} entries")
            df = pd.DataFrame(data)
            
            # Add generation_setting column (derived from model_name and added_turn)
            df['generation_setting'] = f"{model_name}_n_turn_{added_turn}"
            
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
        
        # Create output filename
        safe_model_name = model_name.replace('/', '_')
        output_file = eval_output_dir / f"eval_{dataset_name}_{safe_model_name}_n_turn_{added_turn}.json"
        
        # Check if evaluation already exists
        if output_file.exists():
            print(f"Evaluation already exists: {output_file.name}")
            user_input = input("Do you want to recompute? (y/n): ")
            if user_input.lower() != 'y':
                print("Skipping...")
                continue
        
        # Data Cleaning
        df = remove_invalid_entries(df)
        df, removed_short_ids = remove_short_context_entries(df, dataset_name)
        
        # Save removed unique_ids to file
        removed_file = removed_entries_dir / f"removed_short_{dataset_name}_{safe_model_name}_n_turn_{added_turn}.json"
        with open(removed_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset_name,
                'model': model_name,
                'added_turns': added_turn,
                'removed_count': len(removed_short_ids),
                'removed_unique_ids': removed_short_ids
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved removed unique_ids to: {removed_file.name}")


        # Evaluate
        try:
            evaluator.eval(
                dataset_df=df,
                output_file=output_file,
                metrics=["llm-judge"]
            )
            print(f"Evaluation completed for {json_file.name}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("All evaluations completed!")
    print(f"{'='*70}")
    print(f"Results saved in: {eval_output_dir}")


if __name__ == "__main__":
    main()
