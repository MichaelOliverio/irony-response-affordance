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

            # zero-shot w conversation
            print("\nMale: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='gender_male')

            print("\Female: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='gender_female')

            print("\Boomer: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='generation_boomer')

            print("\Gen Z: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='generation_gen_z')

            print("\Gen X: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='generation_gen_x')

            print("\Gen Y: Using conversation field as input...")
            decoder.zero_shot(input_type='conversation', prompt_strategy='generation_gen_y')

            if dataset_name == 'CSC':
                print("\Friends: Using conversation field as input...")
                decoder.zero_shot(input_type='conversation', prompt_strategy='relationship_friends')

                print("\Colleauges: Using conversation field as input...")
                decoder.zero_shot(input_type='conversation', prompt_strategy='relationship_colleagues')

if __name__ == "__main__":
    main()