"""
Automatic evaluation of generated dialogues using perplexity and LLM-as-a-judge.
"""
import json
import re
from pathlib import Path
import pandas as pd
from src import DialogueEvaluator


def parse_filename(filename):
    """Parse filename to extract dataset name, model name and number of turns"""
    pattern = re.compile(
        r'^(?P<dataset>[^_]+)_(?P<model>.+?)_n_turn_(?P<turn>\d+)\_filtered.json$'
    )
    
    match = pattern.match(filename)
    if match:
        return match.group("dataset"), match.group("model"), int(match.group("turn"))
    else:
        return None


def main():
    """Main function for automatic evaluation."""
    output_dir = Path("output")
    eval_output_dir = output_dir / "evaluations"
    eval_output_dir.mkdir(exist_ok=True)
    
    json_files = list(output_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found in output directory")
        return
    
    print(f"{'='*70}")
    print(f"Automatic Dialogue Evaluation")
    print(f"{'='*70}")
    print(f"Found {len(json_files)} augmented dataset(s) to evaluate\n")
    
    # Initialize evaluator
    evaluator = DialogueEvaluator(
        perplexity_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
        ],
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
            print(f"⚠ Skipping {json_file.name} - doesn't match expected pattern")
            continue
        
        dataset_name, model_name, added_turn = parsed
        print(f"Dataset: {dataset_name}")
        print(f"Augmenter Model: {model_name}")
        print(f"Added Turns: {added_turn}")
        
        # Load data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data)} entries")
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"❌ Error loading {json_file.name}: {e}")
            continue
        
        # Create output filename
        safe_model_name = model_name.replace('/', '_')
        output_file = eval_output_dir / f"eval_{dataset_name}_{safe_model_name}_n_turn_{added_turn}.json"
        
        # Check if evaluation already exists
        if output_file.exists():
            print(f"⚠ Evaluation already exists: {output_file.name}")
            user_input = input("Do you want to recompute? (y/n): ")
            if user_input.lower() != 'y':
                print("Skipping...")
                continue
        
        # Evaluate
        try:
            evaluator.eval(
                dataset_df=df,
                output_file=output_file,
                metrics=["llm-judge"] # ["perplexity", "llm-judge"]
            )
            print(f"✅ Evaluation completed for {json_file.name}")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ All evaluations completed!")
    print(f"{'='*70}")
    print(f"Results saved in: {eval_output_dir}")


if __name__ == "__main__":
    main()
