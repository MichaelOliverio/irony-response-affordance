import logging
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field
from datasets import load_dataset

from .logger_config import get_logger


@dataclass
class FieldMapping:
    """Mapping configuration for a single field"""
    source_name: Optional[str] = None  # Original field name in dataset
    target_name: str = None           # Name in the output
    default_value: Optional[Any] = None  # Default if missing
    
    def __post_init__(self):
        """
        If source_name is None and no default_value provided → use empty string.
        """
        if self.source_name is None and self.default_value is None:
            self.default_value = ""  # default automatico stringa vuota


@dataclass
class DatasetConfig:
    """Configuration class for dataset information"""
    name: str
    source: str  # 'local' or 'huggingface'
    path: str    # file path for local or huggingface identifier
    field_mappings: Dict[str, FieldMapping]  # target_name -> FieldMapping
    id_fields: List[str]  # target field names that compose the unique identifier
    description: Optional[str] = None


class DatasetLoader:
    """
    Loader class for handling different datasets with unified interface.
    Supports local JSON files and Hugging Face datasets.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("DatasetLoader initialized")
        self._setup_dataset_configs()
        
    def _setup_dataset_configs(self):
        """
        Setup predefined dataset configurations
        
        fields:
        - unique_id
        - source
        - context_type
        - context
        - post
        - reply
        - language
        - key_1
        - key_2
        """
        self.dataset_configs = {
            'CSC': DatasetConfig(
                name='CSC',
                source='local',
                path='datasets/CSC/data_full.json',
                field_mappings={
                    'source': FieldMapping(
                        source_name=None,
                        target_name='source',
                        default_value='fictionary'
                    ),
                    'context_type': FieldMapping(
                        source_name='context_type',
                        target_name='context_type'
                    ),
                    'context': FieldMapping(
                        source_name='context_text',
                        target_name='context'
                    ),
                    'post': FieldMapping(
                        source_name=None,
                        target_name='post'
                    ),
                    'reply': FieldMapping(
                        source_name='response_text',
                        target_name='reply'
                    ),
                    'language': FieldMapping(
                        source_name=None,
                        target_name='language',
                        default_value='en'
                    ),
                    'key_1': FieldMapping(
                        source_name='global_context_name',
                        target_name='key_1'
                    ),
                    'key_2': FieldMapping(
                        source_name='global_speaker',
                        target_name='key_2'
                    ),
                },
                id_fields=['key_1', 'key_2'],
                description='Context-based Sarcasm Corpus - local dataset'
            ),
            'MultiPICo': DatasetConfig(
                name='MultiPICo',
                source='huggingface',
                path='Multilingual-Perspectivist-NLU/MultiPICo',
                field_mappings={
                    'source': FieldMapping(
                        source_name='source',
                        target_name='source'
                    ),
                    'context_type': FieldMapping(
                        source_name=None,
                        target_name='context_type'
                    ),
                    'context': FieldMapping(
                        source_name=None,
                        target_name='context'
                    ),
                    'post': FieldMapping(
                        source_name='post',
                        target_name='post'
                    ),
                    'reply': FieldMapping(
                        source_name='reply',
                        target_name='reply'
                    ),
                    'language': FieldMapping(
                        source_name='language',
                        target_name='language'
                    ),
                    'key_1': FieldMapping(
                        source_name='reply_id',
                        target_name='key_1'
                    ),
                    'key_2': FieldMapping(
                        source_name='post_id',
                        target_name='key_2'
                    ),
                },
                id_fields=['key_1', 'key_2'],
                description='Multilingual Perspectivist NLU dataset from Hugging Face'
            )
        }
        self.logger.info(f"Configured {len(self.dataset_configs)} datasets")
    
    def load_dataset(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of the dataset to load ('CSC' or 'MultiPICo')
            **kwargs: Additional arguments for dataset loading
            
        Returns:
            pd.DataFrame: Loaded dataset with only fields of interest
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        self.logger.info(f"Loading dataset: {config.name} from {config.source}")
        
        if config.source == 'local':
            return self._load_local_dataset(config, **kwargs)
        elif config.source == 'huggingface':
            return self._load_huggingface_dataset(config, **kwargs)
        else:
            raise ValueError(f"Unknown source type: {config.source}")
    
    def _load_local_dataset(self, config: DatasetConfig, **kwargs) -> pd.DataFrame:
        """Load dataset from local JSON file"""
        file_path = Path(config.path)
        
        if not file_path.is_absolute():
            # Make path relative to the project root
            current_dir = Path(__file__).parent.parent
            file_path = current_dir / config.path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self.logger.info(f"Loading local dataset from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Apply field mappings
        result_df = self._apply_field_mappings(df, config)
        
        # Add dataset_name field
        result_df['dataset_name'] = config.name
        self.logger.debug(f"Added dataset_name field with value: {config.name}")
        
        # Add unique ID column
        result_df['unique_id'] = self._create_unique_id(result_df, config.id_fields)
        
        # Remove duplicates based on unique_id
        initial_rows = len(result_df)
        result_df = result_df.drop_duplicates(subset=['unique_id'], keep='first')
        duplicates_removed = initial_rows - len(result_df)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows based on unique_id")
        
        self.logger.info(f"Loaded {len(result_df)} rows with fields: {list(result_df.columns)}")
        
        return result_df
    
    def _load_huggingface_dataset(self, config: DatasetConfig, **kwargs) -> pd.DataFrame:
        """Load dataset from Hugging Face"""
        self.logger.info(f"Loading Hugging Face dataset: {config.path}")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(config.path, **kwargs)
            
            # Convert to pandas DataFrame (assuming 'train' split exists)
            if 'train' in dataset:
                df = dataset['train'].to_pandas()
            else:
                # Take the first available split
                split_name = list(dataset.keys())[0]
                df = dataset[split_name].to_pandas()
                self.logger.info(f"Using split: {split_name}")
            
            # Apply field mappings
            result_df = self._apply_field_mappings(df, config)
            
            # Add dataset_name field
            result_df['dataset_name'] = config.name
            self.logger.debug(f"Added dataset_name field with value: {config.name}")
            
            # Add unique ID column
            result_df['unique_id'] = self._create_unique_id(result_df, config.id_fields)
            
            # Remove duplicates based on unique_id
            initial_rows = len(result_df)
            result_df = result_df.drop_duplicates(subset=['unique_id'], keep='first')
            duplicates_removed = initial_rows - len(result_df)
            
            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate rows based on unique_id")
            
            self.logger.info(f"Loaded {len(result_df)} rows with fields: {list(result_df.columns)}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error loading Hugging Face dataset: {e}")
            raise
    
    def _apply_field_mappings(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """
        Apply field mappings to transform the dataframe
        
        Args:
            df: Original dataframe
            config: Dataset configuration with field mappings
            
        Returns:
            pd.DataFrame: Transformed dataframe with renamed fields and default values
        """
        result_df = pd.DataFrame(index=df.index)  # Preserve the index from original df
        
        for target_name, mapping in config.field_mappings.items():
            # Caso: campo da dataset
            if mapping.source_name is not None:
                if mapping.source_name in df.columns:
                    
                    # Se c'è un default → riempi NaN
                    if mapping.default_value is not None:
                        result_df[target_name] = df[mapping.source_name].fillna(mapping.default_value)
                    else:
                        result_df[target_name] = df[mapping.source_name]
                    
                    self.logger.debug(f"Mapped '{mapping.source_name}' -> '{target_name}'")
                
                else:
                    # Campo assente → default obbligatorio (già garantito da __post_init__)
                    result_df[target_name] = mapping.default_value
                    self.logger.warning(
                        f"Field '{mapping.source_name}' not found. Using default value '{mapping.default_value}'"
                    )
            
            # Caso: nessun campo sorgente → default diretto (per tutte le righe)
            else:
                result_df[target_name] = mapping.default_value
                self.logger.debug(f"Created '{target_name}' with default '{mapping.default_value}'")

        return result_df
    
    def _create_unique_id(self, df: pd.DataFrame, id_fields: List[str]) -> pd.Series:
        """Create unique ID by combining specified fields"""
        if len(id_fields) == 1:
            return df[id_fields[0]].astype(str)
        else:
            return df[id_fields].astype(str).agg('_'.join, axis=1)
    
    def get_dataset_info(self, dataset_name: str) -> DatasetConfig:
        """Get configuration information for a dataset"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.dataset_configs.keys())}")
        
        return self.dataset_configs[dataset_name]
    
    def list_available_datasets(self) -> Dict[str, str]:
        """List all available datasets with their descriptions"""
        return {name: config.description or f"{config.name} dataset" 
                for name, config in self.dataset_configs.items()}
    
    def add_custom_dataset(self, config: DatasetConfig):
        """Add a custom dataset configuration"""
        self.dataset_configs[config.name] = config
        self.logger.info(f"Added custom dataset configuration: {config.name}")
    
    def get_sample_data(self, dataset_name: str, n_samples: int = 5) -> pd.DataFrame:
        """Get a sample of the dataset for inspection"""
        df = self.load_dataset(dataset_name)
        return df.head(n_samples)
    
    def create_unified_corpus(self, dataset_names: List[str] = None, output_dir: str = None) -> pd.DataFrame:
        """
        Create a unified corpus by combining multiple datasets
        
        Args:
            dataset_names: List of dataset names to combine (default: all available datasets)
            output_dir: Directory to save the output (default: 'output' in project root)
            
        Returns:
            pd.DataFrame: Unified corpus with all datasets combined
        """
        if dataset_names is None:
            dataset_names = list(self.dataset_configs.keys())
        
        self.logger.info(f"Creating unified corpus from datasets: {dataset_names}")
        
        # Load and combine all datasets
        dataframes = []
        for dataset_name in dataset_names:
            self.logger.info(f"Loading {dataset_name} for unified corpus...")
            df = self.load_dataset(dataset_name)
            dataframes.append(df)
            self.logger.info(f"Added {len(df)} rows from {dataset_name}")
        
        # Concatenate all dataframes
        unified_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined {len(dataframes)} datasets into unified corpus with {len(unified_df)} total rows")
        
        # Check for cross-dataset duplicates (unlikely but possible)
        initial_rows = len(unified_df)
        unified_df = unified_df.drop_duplicates(subset=['unique_id'], keep='first')
        cross_duplicates = initial_rows - len(unified_df)
        
        if cross_duplicates > 0:
            self.logger.warning(f"Removed {cross_duplicates} duplicate rows across datasets")
        
        # Save to output directory
        if output_dir is None:
            current_dir = Path(__file__).parent.parent
            output_dir = current_dir / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_dir / 'unified_corpus.json'
        unified_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        self.logger.info(f"Saved unified corpus to JSON: {json_path}")
        
        # Save as CSV
        csv_path = output_dir / 'unified_corpus.csv'
        unified_df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved unified corpus to CSV: {csv_path}")
        
        # Save summary statistics
        summary_path = output_dir / 'unified_corpus_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("UNIFIED CORPUS SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total rows: {len(unified_df)}\n")
            f.write(f"Total columns: {len(unified_df.columns)}\n")
            f.write(f"Columns: {list(unified_df.columns)}\n\n")
            
            f.write("Dataset distribution:\n")
            dataset_counts = unified_df['dataset_name'].value_counts()
            for dataset, count in dataset_counts.items():
                f.write(f"  - {dataset}: {count} rows ({count/len(unified_df)*100:.2f}%)\n")
            
            f.write("\nLanguage distribution:\n")
            lang_counts = unified_df['language'].value_counts()
            for lang, count in lang_counts.items():
                f.write(f"  - {lang}: {count} rows ({count/len(unified_df)*100:.2f}%)\n")
            
            f.write("\nSource distribution:\n")
            source_counts = unified_df['source'].value_counts()
            for src, count in source_counts.items():
                f.write(f"  - {src}: {count} rows ({count/len(unified_df)*100:.2f}%)\n")
        
        self.logger.info(f"Saved summary statistics to: {summary_path}")
        
        return unified_df