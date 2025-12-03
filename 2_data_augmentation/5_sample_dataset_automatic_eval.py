"""
Automatic evaluation of generated dialogues using perplexity and LLM-as-a-judge.
"""
import json
import re
from pathlib import Path
import pandas as pd
from src import DialogueEvaluator

FILES = [
    'csc_all_instances_for_llm_judge.json',
    'multipico_all_instances_for_llm_judge.json',
]

def main():
    """Main function for automatic evaluation."""
    output_dir = Path("human_annotations/manual_eval_files")
    eval_output_dir = output_dir / "judge_evaluations"
    eval_output_dir.mkdir(exist_ok=True)
    
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
        perplexity_models = [
            #"meta-llama/Llama-3.2-1B-Instruct",
            #"Qwen/Qwen2.5-7B-Instruct",
            #"google/gemma-3-12b-it",
        ],
        judge_models = [
            #"meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-3-12b-it",
        ]
    )
    
    for json_file in json_files:
        print(f"\n{'='*70}")
        print(f"Processing: {json_file.name}")
        print(f"{'='*70}")

        if json_file.name not in FILES:
            print(f"Skipping {json_file.name} - doesn't match expected pattern")
            continue
        
        # Load data
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} entries")
            df = pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue
        
        # Create output filename
        output_file = eval_output_dir / f"judge_scores_{json_file.name}"
        
        # Evaluate
        try:
            evaluator.eval(
                dataset_df=df,
                output_file=output_file,
                metrics=["llm-judge"] # ["perplexity", "llm-judge"]
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
