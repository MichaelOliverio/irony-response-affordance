from src.config import DATASETS, DECODER_SETTINGS, ENCODER_SETTINGS, OUTPUT_DIR
from src.dataset_loader import DatasetLoader 
from src.decoder_handler import DecoderHandler
import argparse

def main():
    do_prompting = False
    do_fine_tuning = True

    for i, dataset_name in enumerate(DATASETS.keys()):
        print(f"\nProcessing dataset: {dataset_name} ({i+1}/{len(DATASETS)})")
        dataset_path = DATASETS[dataset_name]['path']

        print("\nSplitting dataset into train, validation, and test sets...")
        train_set, val_set, test_set  = DatasetLoader.load_aggregated_dataset(dataset_path, verbose=False)

        dataset = {
            'name': dataset_name,
            'train': train_set,
            'val': val_set,
            'test': test_set
        }

        for decoder_model in DECODER_SETTINGS['models']:
            print(f"\n{'='*80}")
            print(f"Using decoder model: {decoder_model}")
            print(f"{'='*80}\n")
            decoder = DecoderHandler(model_name=decoder_model, dataset=dataset)

            if do_prompting:
                # ========================================
                # Prompting
                # ========================================
                print(f"\n{'#'*80}\nPROMPTING\n{'#'*80}\n")
        
                # prompting w context
                print("\nPrompting with context field...")
                decoder.zero_shot(input_type='context')
                decoder.chain_of_thought(input_type='context')

                # prompting w context + response
                print("\nPrompting with context and response fields...")
                decoder.zero_shot(input_type='context_reply')
                decoder.chain_of_thought(input_type='context_reply')

                # prompting w dialogue
                print("\nPrompting with dialogue history...")
                decoder.zero_shot(input_type='conversation')
                decoder.chain_of_thought(input_type='conversation')
            
            if do_fine_tuning:
                # ========================================
                # Fine-tuning
                # ========================================
                print(f"\n{'#'*80}\nFINE-TUNING\n{'#'*80}\n")

                # 1. fine-tuning w context
                print("\nFine-tuning with context field...")
                fine_tuned_model_path = decoder.fine_tuning(input_type='context', force_retrain=False, resume_from_checkpoint=True)
                print("\nGenerating responses after fine-tuning with context...")
                decoder.fine_tuned_inference(input_type='context',fine_tuned_model_path=fine_tuned_model_path)

                # 2. fine-tuning w conversation
                print("\nFine-tuning with dialogue history...")
                fine_tuned_model_path = decoder.fine_tuning(input_type='conversation', force_retrain=False, resume_from_checkpoint=True)
                print("\nGenerating responses after fine-tuning with conversation...")
                decoder.fine_tuned_inference(input_type='conversation',fine_tuned_model_path=fine_tuned_model_path)

        #for encoder_model in ENCODER_SETTINGS['models']:
        #    print(f"\nUsing encoder model: {encoder_model}")

            # fine-tuning w context
        #    print("\nFine-tuning with context field...")
        #    print("\nGenerating responses...")

            # fine-tuning w dialogue
        #    print("\nFine-tuning with dialogue history...")
        #    print("\nGenerating responses...")

if __name__ == "__main__":
    main()