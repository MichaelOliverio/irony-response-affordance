"""
Script to create the final dataset by calculating weighted overall scores
and filtering out entries with score < 0.5

Weighted score formula:
    0.3 * Coherence + 0.1 * Engagingness + 0.3 * Groundedness + 0.3 * Naturalness
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


# Weights for the overall score
WEIGHTS = {
    'Coherence': 0.3,
    'Engagingness': 0.1,
    'Groundedness': 0.3,
    'Naturalness': 0.3
}

# Threshold for filtering
SCORE_THRESHOLD = 0.5

# Paths
OUTPUT_DIR = Path(__file__).parent / "output"
EVALUATIONS_DIR = OUTPUT_DIR / "evaluations"
FINAL_DATASET_DIR = OUTPUT_DIR / "final_dataset"
FINAL_DATASET_DIR.mkdir(exist_ok=True, parents=True)


def calculate_weighted_overall_score(entry: Dict[str, Any]) -> float:
    """
    Calculate the weighted overall score for a single entry.
    
    Args:
        entry: Dictionary containing evaluation scores for all dimensions
        
    Returns:
        Weighted overall score
    """
    score = 0.0
    for dimension, weight in WEIGHTS.items():
        if dimension in entry and 'average_score' in entry[dimension]:
            score += weight * entry[dimension]['average_score']
        else:
            print(f"Warning: Missing {dimension} score for entry {entry.get('unique_id', 'unknown')}")
    
    return score


def load_evaluation_file(filepath: Path) -> Dict[str, Any]:
    """Load evaluation JSON file."""
    print(f"Loading evaluation file: {filepath.name}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_generated_dataset(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the generated dataset and create a dictionary indexed by unique_id.
    
    Args:
        filepath: Path to the generated dataset JSON file
        
    Returns:
        Dictionary mapping unique_id to entry data
    """
    print(f"Loading generated dataset: {filepath.name}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create dictionary indexed by unique_id for fast lookup
    dataset_dict = {}
    for entry in data:
        unique_id = entry.get('unique_id')
        if unique_id:
            dataset_dict[unique_id] = entry
        else:
            print(f"Warning: Entry without unique_id found: Index={entry.get('Index', 'unknown')}")
    
    print(f"  Loaded {len(dataset_dict)} entries")
    return dataset_dict


def process_dataset(eval_filepath: Path, generated_dataset_filepath: Path, dataset_name: str) -> Dict[str, Any]:
    """
    Process a single dataset: calculate weighted scores and filter entries.
    
    Args:
        eval_filepath: Path to the evaluation JSON file
        generated_dataset_filepath: Path to the generated dataset JSON file
        dataset_name: Name of the dataset (MultiPICo or CSC)
        
    Returns:
        Dictionary with statistics and filtered entries
    """
    print(f"\n{'='*80}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*80}")
    
    # Load evaluation data
    eval_data = load_evaluation_file(eval_filepath)
    
    # Load generated dataset
    generated_dataset = load_generated_dataset(generated_dataset_filepath)
    
    # Get the judge model name and entries
    judge_name = list(eval_data['judge'].keys())[0]
    entries = eval_data['judge'][judge_name]
    
    print(f"Judge model: {judge_name}")
    print(f"Total evaluation entries: {len(entries)}")
    print(f"Total generated dataset entries: {len(generated_dataset)}")
    
    # Calculate weighted scores and merge with original data
    entries_with_scores = []
    score_distribution = defaultdict(int)
    missing_entries = []
    
    for eval_entry in entries:
        unique_id = eval_entry.get('unique_id')
        
        # Get the original entry from generated dataset
        original_entry = generated_dataset.get(unique_id)
        
        if not original_entry:
            print(f"Warning: No matching entry in generated dataset for unique_id: {unique_id}")
            missing_entries.append(unique_id)
            continue
        
        # Calculate weighted score
        weighted_score = calculate_weighted_overall_score(eval_entry)
        
        # Store entry with its score (for filtering only, not saving in final dataset)
        entry_with_score = {
            'data': original_entry,
            'score': weighted_score
        }
        
        entries_with_scores.append(entry_with_score)
        
        # Track score distribution
        score_bin = int(weighted_score * 10) / 10  # Round to 1 decimal
        score_distribution[score_bin] += 1
    
    if missing_entries:
        print(f"\nWarning: {len(missing_entries)} entries from evaluation not found in generated dataset")
    
    # Filter entries by threshold (extract only the data, not the scores)
    filtered_entries = [e['data'] for e in entries_with_scores if e['score'] >= SCORE_THRESHOLD]
    removed_entries = [e['data'] for e in entries_with_scores if e['score'] < SCORE_THRESHOLD]
    
    # Statistics
    print(f"\nScore Statistics:")
    print(f"  Total entries: {len(entries_with_scores)}")
    print(f"  Entries >= {SCORE_THRESHOLD}: {len(filtered_entries)} ({len(filtered_entries)/len(entries_with_scores)*100:.2f}%)")
    print(f"  Entries < {SCORE_THRESHOLD}: {len(removed_entries)} ({len(removed_entries)/len(entries_with_scores)*100:.2f}%)")
    
    # Print score distribution
    print(f"\nScore Distribution:")
    for score in sorted(score_distribution.keys()):
        count = score_distribution[score]
        percentage = count / len(entries_with_scores) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {score:.1f}-{score+0.1:.1f}: {count:5d} ({percentage:5.2f}%) {bar}")
    
    # Calculate average scores
    avg_score_all = sum(e['score'] for e in entries_with_scores) / len(entries_with_scores) if entries_with_scores else 0
    avg_score_filtered = sum(e['score'] for e in entries_with_scores if e['score'] >= SCORE_THRESHOLD) / len(filtered_entries) if filtered_entries else 0
    
    print(f"\nAverage Scores:")
    print(f"  All entries: {avg_score_all:.4f}")
    print(f"  Filtered entries (>= {SCORE_THRESHOLD}): {avg_score_filtered:.4f}")
    
    
    # Dimension-wise statistics for filtered entries (using entries_with_scores for calculations)
    print(f"\nDimension-wise Average Scores (Filtered entries):")
    dimension_scores = defaultdict(list)
    for entry_with_score in entries_with_scores:
        if entry_with_score['score'] >= SCORE_THRESHOLD:
            # Get the corresponding evaluation entry
            unique_id = entry_with_score['data']['unique_id']
            eval_entry = next((e for e in entries if e.get('unique_id') == unique_id), None)
            if eval_entry:
                for dimension in WEIGHTS.keys():
                    eval_dim = eval_entry.get(dimension, {})
                    if 'average_score' in eval_dim:
                        dimension_scores[dimension].append(eval_dim['average_score'])
    
    for dimension in WEIGHTS.keys():
        if dimension_scores[dimension]:
            avg = sum(dimension_scores[dimension]) / len(dimension_scores[dimension])
            print(f"  {dimension}: {avg:.4f}")
    
    return {
        'dataset_name': dataset_name,
        'judge_model': judge_name,
        'total_entries': len(entries_with_scores),
        'filtered_entries_count': len(filtered_entries),
        'removed_entries_count': len(removed_entries),
        'threshold': SCORE_THRESHOLD,
        'weights': WEIGHTS,
        'average_score_all': avg_score_all,
        'average_score_filtered': avg_score_filtered,
        'score_distribution': dict(score_distribution),
        'dimension_averages': {dim: sum(scores)/len(scores) if scores else 0 for dim, scores in dimension_scores.items()},
        'filtered_entries': filtered_entries,
        'removed_entries': removed_entries
    }


def save_results(results: Dict[str, Any], output_prefix: str):
    """Save the results to JSON files."""
    dataset_name = results['dataset_name']
    
    # Save filtered dataset (full entries)
    filtered_output = FINAL_DATASET_DIR / f"{output_prefix}_filtered.json"
    filtered_data = {
        'metadata': {
            'dataset_name': results['dataset_name'],
            'judge_model': results['judge_model'],
            'total_entries': results['total_entries'],
            'filtered_entries_count': results['filtered_entries_count'],
            'removed_entries_count': results['removed_entries_count'],
            'threshold': results['threshold'],
            'weights': results['weights'],
            'average_score_all': results['average_score_all'],
            'average_score_filtered': results['average_score_filtered'],
            'dimension_averages': results['dimension_averages']
        },
        'entries': results['filtered_entries']
    }
    
    with open(filtered_output, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved filtered dataset to: {filtered_output}")
    
    # Save removed entries (for analysis)
    removed_output = FINAL_DATASET_DIR / f"{output_prefix}_removed.json"
    removed_data = {
        'metadata': {
            'dataset_name': results['dataset_name'],
            'judge_model': results['judge_model'],
            'removed_entries_count': results['removed_entries_count'],
            'threshold': results['threshold'],
            'weights': results['weights']
        },
        'removed_entries': results['removed_entries']
    }
    
    with open(removed_output, 'w', encoding='utf-8') as f:
        json.dump(removed_data, f, indent=2, ensure_ascii=False)
    print(f"Saved removed entries to: {removed_output}")
    
    # Save summary statistics
    summary_output = FINAL_DATASET_DIR / f"{output_prefix}_summary.json"
    summary_data = {
        'dataset_name': results['dataset_name'],
        'judge_model': results['judge_model'],
        'total_entries': results['total_entries'],
        'filtered_entries_count': results['filtered_entries_count'],
        'removed_entries_count': results['removed_entries_count'],
        'threshold': results['threshold'],
        'weights': results['weights'],
        'average_score_all': results['average_score_all'],
        'average_score_filtered': results['average_score_filtered'],
        'score_distribution': results['score_distribution'],
        'dimension_averages': results['dimension_averages']
    }
    
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved summary statistics to: {summary_output}")


def main():
    """Main function to process all datasets."""
    print("="*80)
    print("Creating Final Filtered Dataset")
    print("="*80)
    print(f"\nWeighted Overall Score Formula:")
    print(f"  {WEIGHTS['Coherence']} × Coherence + {WEIGHTS['Engagingness']} × Engagingness + {WEIGHTS['Groundedness']} × Groundedness + {WEIGHTS['Naturalness']} × Naturalness")
    print(f"\nScore Threshold: {SCORE_THRESHOLD}")
    print(f"Entries with weighted_overall_score < {SCORE_THRESHOLD} will be removed")
    
    # Dataset configurations
    datasets = [
        {
            'name': 'MultiPICo',
            'eval_file': 'prompt9_eval_MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2.json',
            'generated_file': 'MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2.json',
            'output_prefix': 'MultiPICo_meta-llama_Llama-3.1-8B-Instruct_n_turn_2'
        },
        {
            'name': 'CSC',
            'eval_file': 'prompt9_eval_CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2.json',
            'generated_file': 'CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2.json',
            'output_prefix': 'CSC_meta-llama_Llama-3.1-8B-Instruct_n_turn_2'
        }
    ]
    
    all_results = []
    
    # Process each dataset
    for dataset_config in datasets:
        eval_filepath = EVALUATIONS_DIR / dataset_config['eval_file']
        generated_filepath = OUTPUT_DIR / dataset_config['generated_file']
        
        if not eval_filepath.exists():
            print(f"\nWARNING: Evaluation file not found: {eval_filepath}")
            continue
        
        if not generated_filepath.exists():
            print(f"\nWARNING: Generated dataset file not found: {generated_filepath}")
            continue
        
        # Process dataset
        results = process_dataset(eval_filepath, generated_filepath, dataset_config['name'])
        all_results.append(results)
        
        # Save results
        save_results(results, dataset_config['output_prefix'])
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("Overall Summary")
    print(f"{'='*80}")
    
    total_entries = sum(r['total_entries'] for r in all_results)
    total_filtered = sum(r['filtered_entries_count'] for r in all_results)
    total_removed = sum(r['removed_entries_count'] for r in all_results)
    
    print(f"\nTotal across all datasets:")
    print(f"  Total entries: {total_entries}")
    print(f"  Filtered entries (>= {SCORE_THRESHOLD}): {total_filtered} ({total_filtered/total_entries*100:.2f}%)")
    print(f"  Removed entries (< {SCORE_THRESHOLD}): {total_removed} ({total_removed/total_entries*100:.2f}%)")
    
    print(f"\nAll results saved to: {FINAL_DATASET_DIR}")
    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
