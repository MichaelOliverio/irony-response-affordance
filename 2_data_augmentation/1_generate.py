from src import UnifiedDatasetLoader, DialogueDataAugmentor

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
        #"CSC": {
        #    "input_type": "contextual",
        #    "output_type": "multi-turn",
        #    "languages": ["en"],
        #}
    }

    for dataset_name, params in datasets.items():
        print(f"\n{'='*70}")
        print(f"=== Processing dataset: {dataset_name} ===")
        print(f"{'='*70}")
        
        languages = params["languages"]

        # 1. Caricare il dataset
        print("\n[1/3] Caricamento dataset...")
        df = UnifiedDatasetLoader.load_test_set(dataset_name, 'en') # UnifiedDatasetLoader.load_dataset([dataset_name], languages)

        print(f" Usando {len(df)} righe per testing")

        # 2. Data augmentation
        print("\n[2/3] Data augmentation...")
        augmented_files = DialogueDataAugmentor.generate_dialogues_for_dataset(
            dataset_df=df,
            dataset_name=dataset_name,
        )

        print(f"\n Data augmentation completata")
        for model_name, file_path in augmented_files.items():
            print(f"  - {model_name}: {file_path}")

    print(f"\n{'='*70}")
    print(" Pipeline completata")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()