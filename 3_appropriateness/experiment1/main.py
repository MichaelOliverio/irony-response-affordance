from src.config import DATASETS, DECODER_SETTINGS, ENCODER_SETTINGS, OUTPUT_DIR
from src.dataset_loader import DatasetLoader 
from src.decoder_handler import DecoderHandler

def main():
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
            print(f"\nUsing decoder model: {decoder_model}")
            decoder = DecoderHandler(model_name=decoder_model, dataset=dataset)

            # baseline1: zero-shot w context
            print("\nBaseline 1: Using context field as input...")
            decoder.zero_shot(input_type='context')

            continue

            # baseline2: zero-shot w context * response
            print("\nBaseline 2: Using context and response fields as input...")
            decoder.zero_shot(input_type='context_reply')

            # prompting w dialogue
            print("\nZero-shot with dialogue history...")
            decoder.zero_shot(input_type='conversation')

            # fine-tuning w context
            #print("\nFine-tuning with context field...")
            #print("\nGenerating responses...")



            # fine-tuning w dialogue
            #print("\nFine-tuning with dialogue history...")
            #print("\nGenerating responses...")

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