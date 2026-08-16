from src.config import DATASETS, DECODER_SETTINGS, ENCODER_SETTINGS, OUTPUT_DIR
from src.dataset_loader import DatasetLoader 
from src.decoder_handler import DecoderHandler
from src.encoder_handler import EncoderHandler
import argparse

def main():
    do_prompting = True
    do_fine_tuning = False
    do_encoder = False

    for i, dataset_name in enumerate(DATASETS.keys()):
        print(f"\nProcessing dataset: {dataset_name} ({i+1}/{len(DATASETS)})")
        dataset_path = DATASETS[dataset_name]['path']

        print("\nSplitting dataset into train, validation, and test sets...")
        all_entries = DatasetLoader.load_cleaned_dataset(dataset_path)
        human_annotated_ids = DatasetLoader.load_human_annotated_ids(
            dataset_name
        )
        non_human_entries = [
            entry
            for entry in all_entries
            if str(entry['unique_id']) not in human_annotated_ids
        ]
        train_set, val_set, test_set = DatasetLoader.balanced_split(
            non_human_entries
        )
        human_test_set = DatasetLoader.build_human_evaluation_set(
            all_entries, dataset_name
        )

        dataset = {
            'name': dataset_name,
            'train': train_set,
            'val': val_set,
            'test': test_set,
            'human_test': human_test_set,
        }

        # ========================================
        # DECODER MODELS
        # ========================================
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
                decoder.zero_shot(input_type='last_u1_message')
                decoder.chain_of_thought(input_type='last_u1_message')

                # prompting w context + response
                #print("\nPrompting with context and response fields...")
                #decoder.zero_shot(input_type='context_reply')
                #decoder.chain_of_thought(input_type='context_reply')

                # prompting w dialogue
                #print("\nPrompting with dialogue history...")
                #decoder.zero_shot(input_type='conversation')
                #decoder.chain_of_thought(input_type='conversation')
            
            if do_fine_tuning:
                # ========================================
                # Fine-tuning
                # ========================================
                print(f"\n{'#'*80}\nFINE-TUNING\n{'#'*80}\n")

                # 1. fine-tuning w last_u1_message
                print("\nFine-tuning with last_u1_message field...")
                fine_tuned_model_path = decoder.fine_tuning(input_type='last_u1_message', force_retrain=False, resume_from_checkpoint=True)
                print("\nGenerating responses after fine-tuning with last_u1_message...")
                decoder.fine_tuned_inference(input_type='last_u1_message',fine_tuned_model_path=fine_tuned_model_path)

                # 2. fine-tuning w conversation
                print("\nFine-tuning with dialogue history...")
                fine_tuned_model_path = decoder.fine_tuning(input_type='conversation', force_retrain=False, resume_from_checkpoint=True)
                print("\nGenerating responses after fine-tuning with conversation...")
                decoder.fine_tuned_inference(input_type='conversation',fine_tuned_model_path=fine_tuned_model_path)

        # ========================================
        # ENCODER MODELS
        # ========================================
        if do_encoder:
            for encoder_model in ENCODER_SETTINGS['models']:
                print(f"\n{'='*80}")
                print(f"Using encoder model: {encoder_model}")
                print(f"{'='*80}\n")
                
                encoder = EncoderHandler(model_name=encoder_model, dataset=dataset)

                # ========================================
                # Fine-tuning with context (single-turn)
                # ========================================
                print(f"\n{'#'*80}\nENCODER FINE-TUNING - SINGLE TURN (context_reply)\n{'#'*80}\n")
                fine_tuned_model_path = encoder.train(input_type='context_reply', force_retrain=False)
                print("\nGenerating predictions after fine-tuning with context_reply...")
                encoder.predict(input_type='context_reply', fine_tuned_model_path=fine_tuned_model_path)
                encoder.predict(
                    input_type='context_reply',
                    fine_tuned_model_path=fine_tuned_model_path,
                    evaluation_split='human_test',
                )

                # ========================================
                # Fine-tuning with context (single-turn)
                # ========================================
                print(f"\n{'#'*80}\nENCODER FINE-TUNING - SINGLE TURN (last message of u1)\n{'#'*80}\n")
                fine_tuned_model_path = encoder.train(input_type='last_u1_message', force_retrain=False)
                print("\nGenerating predictions after fine-tuning with last message of u1...")
                encoder.predict(input_type='last_u1_message', fine_tuned_model_path=fine_tuned_model_path)
                encoder.predict(
                    input_type='last_u1_message',
                    fine_tuned_model_path=fine_tuned_model_path,
                    evaluation_split='human_test',
                )

                # ========================================
                # Fine-tuning with conversation (multi-turn)
                # ========================================
                print(f"\n{'#'*80}\nENCODER FINE-TUNING - MULTI TURN (CONVERSATION)\n{'#'*80}\n")
                fine_tuned_model_path = encoder.train(input_type='conversation', force_retrain=False)
                print("\nGenerating predictions after fine-tuning with conversation...")
                encoder.predict(input_type='conversation', fine_tuned_model_path=fine_tuned_model_path)
                encoder.predict(
                    input_type='conversation',
                    fine_tuned_model_path=fine_tuned_model_path,
                    evaluation_split='human_test',
                )

if __name__ == "__main__":
    main()
