import logging
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from datasets import load_dataset

from .logger_config import get_logger
from .dataset_merger import DatasetMerger


@dataclass
class DatasetConfig:
    """Configuration class for dataset information"""
    name: str
    source: str  # 'local' or 'huggingface'
    path: str    # file path for local or huggingface identifier
    field_mappings: Dict[str, Any]  # Placeholder, not used anymore
    id_fields: List[str]  # field names that compose the unique identifier
    description: Optional[str] = None


class DatasetLoader:
    """
    Loader class for handling different datasets with unified interface.
    Supports local JSON files and Hugging Face datasets.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("DatasetLoader initialized")
        self.dataset_merger = DatasetMerger()
        self._setup_dataset_configs()
        
    def _setup_dataset_configs(self):
        """Setup predefined dataset configurations"""
        self.dataset_configs = {
            'CSC': DatasetConfig(
                name='CSC',
                source='local',
                path='datasets/CSC/data_full.json',
                field_mappings={},
                id_fields=['global_context_name', 'global_speaker'],
                description='Conversational Sarcasm Corpus - local dataset'
            ),
            'MultiPICo': DatasetConfig(
                name='MultiPICo',
                source='huggingface',
                path='Multilingual-Perspectivist-NLU/MultiPICo',
                field_mappings={},
                id_fields=['reply_id', 'post_id'],
                description='Multilingual Perspectivist NLU dataset from Hugging Face'
            )
        }
        self.logger.info(f"Configured {len(self.dataset_configs)} datasets")
    
    def list_available_datasets(self) -> Dict[str, str]:
        """List all available datasets with their descriptions"""
        return {name: config.description or f"{config.name} dataset" 
                for name, config in self.dataset_configs.items()}
    
    def load_dataset_with_annotations(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Load dataset with structured annotations (no aggregation, preserves all annotation details)
        
        Args:
            dataset_name: Name of the dataset to load ('CSC' or 'MultiPICo')
            
        Returns:
            List[Dict]: List of structured annotation dictionaries
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        self.logger.info(f"Loading dataset with annotations: {config.name}")
        
        # Load raw data
        df_raw = None
        if config.source == 'local':
            file_path = Path(config.path)
            if not file_path.is_absolute():
                current_dir = Path(__file__).parent.parent
                file_path = current_dir / config.path
            
            self.logger.info(f"Loading raw data from: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df_raw = pd.DataFrame(data)
            
        elif config.source == 'huggingface':
            self.logger.info(f"Loading raw data from Hugging Face: {config.path}")
            dataset = load_dataset(config.path)
            if 'train' in dataset:
                df_raw = dataset['train'].to_pandas()
                # Fallback to first split if 'train' not available (MultiPICo case)
        
        if df_raw is None:
            raise ValueError(f"Failed to load data for dataset '{dataset_name}'")

        # Process annotations based on dataset type
        if dataset_name == 'CSC':
            structured_data = self.dataset_merger.process_csc_annotations(df_raw)
        elif dataset_name == 'MultiPICo':
            structured_data = self.dataset_merger.process_multipico_annotations(df_raw)
        else:
            raise ValueError(f"Annotation processing not implemented for {dataset_name}")
        
        return structured_data
    
    def create_unified_corpus(self, dataset_names: List[str] = None, output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Create a unified corpus with structured annotations
        
        Args:
            dataset_names: List of dataset names to combine (default: all available datasets)
            output_dir: Directory to save the output (default: 'output' in project root)
            
        Returns:
            List[Dict]: Unified corpus with structured annotations
        """
        if dataset_names is None:
            dataset_names = list(self.dataset_configs.keys())
        
        self.logger.info(f"Creating unified corpus with annotations from datasets: {dataset_names}")
        
        # Load and combine all datasets
        all_data = []
        for dataset_name in dataset_names:
            self.logger.info(f"Loading {dataset_name} with annotations...")
            data = self.load_dataset_with_annotations(dataset_name)
            all_data.extend(data)
            self.logger.info(f"Added {len(data)} entries from {dataset_name}")
        
        self.logger.info(f"Combined {len(dataset_names)} datasets into unified corpus with {len(all_data)} total entries")
        
        # Save to output directory
        if output_dir is None:
            current_dir = Path(__file__).parent.parent
            output_dir = current_dir / 'output'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON with structured annotations
        json_path = output_dir / 'unified_corpus.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved unified corpus with annotations to: {json_path}")

        # Save a smaller sample JSON with 10 random entries
        sample_size = min(10, len(all_data))  # per evitare errori se hai meno di 10 entry
        sample_data = pd.DataFrame(all_data).sample(sample_size, random_state=42).to_dict(orient='records')

        sample_json_path = output_dir / 'unified_corpus_sample_10.json'
        with open(sample_json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved 10-entry sample to: {sample_json_path}")

        # Create CSV version with expanded annotations
        self._create_csv_with_expanded_annotations(all_data, output_dir)
        
        # Create benchmark datasets
        self._create_benchmark_datasets(all_data, output_dir)
        
        return all_data
    
    def _create_csv_with_expanded_annotations(self, data: List[Dict[str, Any]], output_dir: Path) -> None:
        """
        Create CSV version of the unified corpus with expanded annotation columns
        
        Args:
            data: Unified corpus data
            output_dir: Output directory path
        """
        self.logger.info("Creating CSV with expanded annotations...")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Expand the annotations dictionary into separate columns
        annotations_df = pd.json_normalize(df['annotations'])
        annotations_df.columns = ['annotations.' + col for col in annotations_df.columns]
        
        # Concatenate with the original dataframe
        df_expanded = pd.concat([df.drop('annotations', axis=1), annotations_df], axis=1)
        
        # Save as CSV
        csv_path = output_dir / 'unified_corpus.csv'
        df_expanded.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved CSV with expanded annotations to: {csv_path}")
    
    def _create_benchmark_datasets(self, data: List[Dict[str, Any]], output_dir: Path) -> None:
        """
        Create benchmark datasets for MultiPICo and CSC
        
        Args:
            data: Unified corpus data
            output_dir: Output directory path
        """
        self.logger.info("Creating benchmark datasets...")
        
        # Convert to DataFrame with expanded annotations
        df = pd.DataFrame(data)
        annotations_df = pd.json_normalize(df['annotations'])
        annotations_df.columns = ['annotations.' + col for col in annotations_df.columns]
        df_expanded = pd.concat([df.drop('annotations', axis=1), annotations_df], axis=1)
        
        # Create MultiPICo benchmark
        self._create_multipico_benchmark(df_expanded, output_dir)
        
        # Create CSC benchmark
        self._create_csc_benchmark(df_expanded, output_dir)
    
    def _create_multipico_benchmark(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Create MultiPICo benchmark: 500 ironical + 500 non-ironical English samples
        
        Args:
            df: DataFrame with expanded annotations
            output_dir: Output directory path
        """
        self.logger.info("Creating MultiPICo benchmark...")
        
        # Filter MultiPICo English data
        df_multipico = df[(df['dataset_name'] == 'MultiPICo') & (df['language'] == 'en')]
        self.logger.info(f"Found {len(df_multipico)} MultiPICo English entries")
        
        if len(df_multipico) == 0:
            self.logger.warning("No MultiPICo English data found, skipping benchmark creation")
            return
        
        # Sample 500 ironical and 500 non-ironical (or as many as available)
        df_ironical = df_multipico[df_multipico['annotations.irony_appropriateness_label'] == 1]
        df_non_ironical = df_multipico[df_multipico['annotations.irony_appropriateness_label'] == 0]
        
        n_ironical = min(500, len(df_ironical))
        n_non_ironical = min(500, len(df_non_ironical))
        
        self.logger.info(f"Sampling {n_ironical} ironical and {n_non_ironical} non-ironical instances")
        
        df_ironical_sampled = df_ironical.sample(n=n_ironical, random_state=42) if n_ironical > 0 else pd.DataFrame()
        df_non_ironical_sampled = df_non_ironical.sample(n=n_non_ironical, random_state=42) if n_non_ironical > 0 else pd.DataFrame()
        
        df_sampled = pd.concat([df_ironical_sampled, df_non_ironical_sampled])
        df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save as CSV and JSON
        csv_path = output_dir / 'multipico_benchmark.csv'
        json_path = output_dir / 'multipico_benchmark.json'
        
        df_sampled.to_csv(csv_path, index=False, encoding='utf-8')
        df_sampled.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        self.logger.info(f"Saved MultiPICo benchmark ({len(df_sampled)} entries) to: {csv_path} and {json_path}")
    
    def _create_csc_benchmark(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Create CSC benchmark: 500 sarcastic + 500 non-sarcastic English samples
        
        Args:
            df: DataFrame with expanded annotations
            output_dir: Output directory path
        """
        self.logger.info("Creating CSC benchmark...")
        
        # Filter CSC English data
        df_csc = df[(df['dataset_name'] == 'CSC') & (df['language'] == 'en')]
        self.logger.info(f"Found {len(df_csc)} CSC English entries")
        
        if len(df_csc) == 0:
            self.logger.warning("No CSC English data found, skipping benchmark creation")
            return
        
        # Sample 500 sarcastic and 500 non-sarcastic (or as many as available)
        df_sarcastic = df_csc[df_csc['annotations.sarcasm_appropriateness_label'] == 1]
        df_non_sarcastic = df_csc[df_csc['annotations.sarcasm_appropriateness_label'] == 0]
        
        n_sarcastic = min(500, len(df_sarcastic))
        n_non_sarcastic = min(500, len(df_non_sarcastic))
        
        self.logger.info(f"Sampling {n_sarcastic} sarcastic and {n_non_sarcastic} non-sarcastic instances")
        
        df_sarcastic_sampled = df_sarcastic.sample(n=n_sarcastic, random_state=42) if n_sarcastic > 0 else pd.DataFrame()
        df_non_sarcastic_sampled = df_non_sarcastic.sample(n=n_non_sarcastic, random_state=42) if n_non_sarcastic > 0 else pd.DataFrame()
        
        df_sampled = pd.concat([df_sarcastic_sampled, df_non_sarcastic_sampled])
        df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save as CSV and JSON
        csv_path = output_dir / 'csc_benchmark.csv'
        json_path = output_dir / 'csc_benchmark.json'
        
        df_sampled.to_csv(csv_path, index=False, encoding='utf-8')
        df_sampled.to_json(json_path, orient='records', indent=2, force_ascii=False)
        
        self.logger.info(f"Saved CSC benchmark ({len(df_sampled)} entries) to: {csv_path} and {json_path}")