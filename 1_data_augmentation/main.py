import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

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
    
    # Load CSC dataset
    logger.info("="*50)
    logger.info("Loading CSC dataset...")
    try:
        csc_data = loader.load_dataset('CSC')
        logger.info(f"CSC dataset loaded successfully with {len(csc_data)} rows")
        logger.info(f"Columns: {list(csc_data.columns)}")
        logger.info("First 3 rows:")
        logger.info(f"\n{csc_data.head(3).to_string()}")
    except Exception as e:
        logger.error(f"Error loading CSC dataset: {e}", exc_info=True)
    
    # Get dataset info
    logger.info("="*50)
    logger.info("CSC Dataset Configuration:")
    csc_config = loader.get_dataset_info('CSC')
    logger.info(f"  Name: {csc_config.name}")
    logger.info(f"  Source: {csc_config.source}")
    logger.info(f"  Path: {csc_config.path}")
    logger.info(f"  ID fields: {csc_config.id_fields}")
    logger.info("  Field mappings:")
    for target_name, mapping in csc_config.field_mappings.items():
        if mapping.source_name:
            logger.info(f"    {mapping.source_name} -> {target_name}")
        else:
            logger.info(f"    {target_name} (default: {mapping.default_value})")
    
    # Try to load MultiPICo dataset (might fail if not available)
    logger.info("="*50)
    logger.info("Loading MultiPICo dataset...")
    try:
        multipico_data = loader.load_dataset('MultiPICo')
        logger.info(f"MultiPICo dataset loaded successfully with {len(multipico_data)} rows")
        logger.info(f"Columns: {list(multipico_data.columns)}")
        logger.info("First 3 rows:")
        logger.info(f"\n{multipico_data.head(3).to_string()}")
    except Exception as e:
        logger.error(f"Error loading MultiPICo dataset: {e}", exc_info=True)
        logger.warning("Note: You might need to install 'datasets' library: pip install datasets")
    
    logger.info("="*50)
    logger.info("Data augmentation pipeline completed")
    logger.info("="*50)
    
    # Create unified corpus
    logger.info("="*50)
    logger.info("Creating unified corpus...")
    try:
        unified_corpus = loader.create_unified_corpus()
        logger.info(f"Unified corpus created successfully with {len(unified_corpus)} total rows")
        logger.info(f"Columns: {list(unified_corpus.columns)}")
        logger.info("\nDataset distribution:")
        for dataset, count in unified_corpus['dataset_name'].value_counts().items():
            logger.info(f"  - {dataset}: {count} rows")
    except Exception as e:
        logger.error(f"Error creating unified corpus: {e}", exc_info=True)
    
    logger.info("="*50)
    logger.info("All operations completed successfully")
    logger.info("="*50)




if __name__ == "__main__":
    main()