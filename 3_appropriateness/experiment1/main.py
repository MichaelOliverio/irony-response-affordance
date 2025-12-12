from src.config import DATASETS, DECODER_MODELS, ENCODER_MODELS, OUTPUT_DIR
 

def main():
    for i, dataset_name in enumerate(DATASETS.keys()):
        print(f"Dataset {i+1}: {dataset_name}")

        # dataset splitting (70, 15, 15) with balanced classes
        print("  Splitting dataset into train, validation, and test sets...")

        for decoder_model in DECODER_MODELS:
            print(f"    Using decoder model: {decoder_model}")

            # baseline1: zero-shot w context
            print("  Baseline 1: Using context field as input...")

            # baseline2: zero-shot w context * response
            print("  Baseline 2: Using context and response fields as input...")

            # fine-tuning w context
            print("  Fine-tuning with context field...")
            print("  Generating responses...")

            # prompting w dialogue
            print("  Prompting with dialogue history...")
            print("  Generating responses...")

            # fine-tuning w dialogue
            print("  Fine-tuning with dialogue history...")
            print("  Generating responses...")

        for encoder_model in ENCODER_MODELS:
            print(f"    Using encoder model: {encoder_model}")

            # fine-tuning w context
            print("  Fine-tuning with context field...")
            print("  Generating responses...")

            # fine-tuning w dialogue
            print("  Fine-tuning with dialogue history...")
            print("  Generating responses...")

if __name__ == "__main__":
    main()