from src import UnifiedDatasetLoader, DialogueDataAugmentor, DataAugmentationEvaluator

def main():
    """
    Pipeline completa per data augmentation e valutazione.
    """
    datasets = {
        "MultiPICo": {
            "input_type": "single-turn",
            "output_type": "multi-turn",
            "languages": ["en"],
        },
        "CSC": {
            "input_type": "contextual",
            "output_type": "multi-turn",
            "languages": ["en"],
        }
    }

    for dataset_name, params in datasets.items():
        print(f"\n{'='*70}")
        print(f"=== Processing dataset: {dataset_name} ===")
        print(f"{'='*70}")
        
        languages = params["languages"]
        input_type = params["input_type"]
        output_type = params["output_type"]

        # 1. Caricare il dataset
        print("\n[1/3] Caricamento dataset...")
        df = UnifiedDatasetLoader.load_dataset([dataset_name], languages)
        
        # Limita il numero di esempi per testing (rimuovere in produzione)
        # df = df.head(10)
        # print(f"  Usando solo {len(df)} esempi per testing")

        # 2. Data augmentation
        print("\n[2/3] Data augmentation...")
        augmented_files = DialogueDataAugmentor.augment_data(
            df=df,
            input_type=input_type,
            output_type=output_type,
        )

        print(f"\n Data augmentation completata")
        for model_name, file_path in augmented_files.items():
            print(f"  - {model_name}: {file_path}")

        # 3. Valutazione
        print("\n[3/3] Valutazione...")
        evaluation_results = DataAugmentationEvaluator.evaluate_all(
            augmented_files=augmented_files,
            run_perplexity=False,  # TODO: Implementare
            run_llm_judge=False,   # TODO: Implementare
        )

        if evaluation_results:
            print("\n Risultati Valutazione:")
            for eval_type, results in evaluation_results.items():
                print(f"\n{eval_type.upper()}:")
                for model_name, scores in results.items():
                    print(f"  {model_name}:")
                    for metric, value in scores.items():
                        print(f"    - {metric}: {value}")

    print(f"\n{'='*70}")
    print(" Pipeline completata")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()