from src import UnifiedDatasetLoader
from src import DialogueDataAugmentor
from src import DataAugmentationEvaluator

def main():
    datasets = {
        "MultiPICo": {
            "input_type": "single-turn",
            "output_type": "multi-turn",
            "languages": ["en"],
        } ,
        "CSC": {
            "input_type": "contextual",
            "output_type": "multi-turn",
            "languages": ["en"],
        }
    }


    for dataset_name, params in datasets.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        languages = params["languages"]
        input_type = params["input_type"]
        output_type = params["output_type"]

        # 1. caricare il dataset
        df = UnifiedDatasetLoader.load_dataset([dataset_name], languages)

        # 2. fare data augmentation
        augmented_data = DialogueDataAugmentor.augment_data(df, input_type, output_type)

        # 4. calcolare la perplexity
        perplexity = DataAugmentationEvaluator.perplexity_eval(augmented_data)

        # 5. llm judge
        llm_judge_results = DataAugmentationEvaluator.llm_judge_eval(augmented_data)

        print(f"Perplexity: {perplexity}")
        print(f"LLM Judge Results: {llm_judge_results}")


if __name__ == "__main__":
    main()