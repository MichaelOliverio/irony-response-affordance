from src import DatasetLoader
from src.logger_config import setup_logger

def main():
    # Setup logger
    logger = setup_logger('main')
    logger.info("="*50)
    logger.info("Starting data augmentation pipeline")
    logger.info("="*50)
    
    # Initialize the dataset loader
    logger.info("Initializing DatasetLoader")
    loader = DatasetLoader()
    
    # List available datasets
    logger.info("Available datasets:")
    for name, description in loader.list_available_datasets().items():
        logger.info(f"  - {name}: {description}")
    
    # Create unified corpus with structured annotations
    logger.info("="*50)
    logger.info("Creating unified corpus with structured annotations...")
    try:
        unified_with_annotations = loader.create_unified_corpus()
        logger.info(f"Unified corpus with annotations created successfully with {len(unified_with_annotations)} entries")
        logger.info(f"Sample entry structure:")
        if len(unified_with_annotations) > 0:
            sample = unified_with_annotations[0]
            logger.info(f"  - Dataset: {sample['dataset_name']}")
            logger.info(f"  - Unique ID: {sample['unique_id']}")
            logger.info(f"  - Number of observer annotations: {len(sample['annotations']['observers_annotations'])}")
            logger.info(f"  - Has speaker annotation: {sample['annotations']['speaker_annotation'] is not None}")
    except Exception as e:
        logger.error(f"Error creating unified corpus with annotations: {e}", exc_info=True)
    
    logger.info("="*50)
    logger.info("All operations completed successfully")
    logger.info("="*50)




if __name__ == "__main__":
    main()