import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .logger_config import get_logger


class DatasetMerger:
    """
    Process and aggregate annotations from different datasets
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def compute_majority_vote(self, series: pd.Series) -> int | str:
        """
        Calcola il voto maggioritario da una serie di label.
        
        Args:
            series: Serie pandas contenente i label (0 o 1)
            
        Returns:
            int | str: 0, 1 (1 anche in caso di parità)
        """
        count_0 = (series == 0).sum()
        count_1 = (series == 1).sum()
        
        if count_0 > count_1:
            return 0
        elif count_1 >= count_0:
            return 1
    
    def process_multipico_annotations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process MultiPICo dataset to create structured annotations
        
        Args:
            df: MultiPICo dataframe with all annotations
            
        Returns:
            List of structured annotation dictionaries
        """
        self.logger.info("Processing MultiPICo annotations...")
        
        # Group by unique conversation (reply_id + post_id)
        grouped = df.groupby(['reply_id', 'post_id'])
        
        result = []
        for (reply_id, post_id), group in grouped:
            unique_id = f"{reply_id}_{post_id}"
            
            # Get base information (same for all rows in group)
            first_row = group.iloc[0]
            
            # Calculate aggregate irony scores
            irony_labels = group['label']
            majority_irony = self.compute_majority_vote(irony_labels)
            #average_irony_score = irony_labels.mean()
            
            # Build observer annotations (all rows are observers in MultiPICo)
            observers_annotations = []
            for _, row in group.iterrows():
                observer = {
                    'annotator_id': str(row['annotator_id']),
                    'irony_appropriateness_label': int(row['label']), # working hypothesis
                    'irony_label': int(row['label']),
                    'irony_score': None,
                    'sarcasm_appropriateness_label': None,  # MultiPICo doesn't have sarcasm
                    'sarcasm_label': None,  # MultiPICo doesn't have sarcasm
                    'sarcasm_score': None,
                    'language_variety': str(row['language_variety']) if pd.notna(row['language_variety']) else '',
                    'age': int(row['Age']) if pd.notna(row['Age']) else None,
                    'gender': str(row['Gender']) if pd.notna(row['Gender']) else None,
                    'ethnicity_simplified': str(row['Ethnicity simplified']) if pd.notna(row['Ethnicity simplified']) else None,
                    'country_of_birth': str(row['Country of birth']) if pd.notna(row['Country of birth']) else None,
                    'country_of_residence': str(row['Country of residence']) if pd.notna(row['Country of residence']) else None,
                    'nationality': str(row['Nationality']) if pd.notna(row['Nationality']) else None,
                    'student_status': str(row['Student status']) if pd.notna(row['Student status']) else None,
                    'employment_status': str(row['Employment status']) if pd.notna(row['Employment status']) else None,
                    'generation': str(row['Generation']) if pd.notna(row['Generation']) else None,
                    'generation_aggregated': str(row['GenerationAggregated']) if pd.notna(row['GenerationAggregated']) else None,
                    'employed': str(row['Employed']) if pd.notna(row['Employed']) else None
                }
                observers_annotations.append(observer)
            
            # Create structured entry
            entry = {
                'source': str(first_row['source']) if pd.notna(first_row['source']) else None,
                'context_type': None,  # Not available in MultiPICo
                'context': None,  # Not available in MultiPICo
                'post': str(first_row['post']) if pd.notna(first_row['post']) else '',
                'reply': str(first_row['reply']) if pd.notna(first_row['reply']) else '',
                'language': str(first_row['language']) if pd.notna(first_row['language']) else None,
                'key_1': str(reply_id),
                'key_2': str(post_id),
                'dataset_name': 'MultiPICo',
                'unique_id': unique_id,
                'annotations': {
                    'irony_appropriateness_label': majority_irony, # working hypothesis
                    'aggregate_irony_label': majority_irony,
                    'average_irony_score': None, #float(average_irony_score),
                    'sarcasm_appropriateness_label': None,
                    'aggregate_sarcasm_label': None,
                    'average_sarcasm_score': None,
                    'speaker_annotation': None,  # MultiPICo doesn't have speaker annotations
                    'observers_annotations': observers_annotations
                }
            }
            
            result.append(entry)
        
        self.logger.info(f"Processed {len(result)} unique conversations from MultiPICo")
        return result
    
    def process_csc_annotations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process CSC dataset to create structured annotations
        
        Args:
            df: CSC dataframe with all annotations
            
        Returns:
            List of structured annotation dictionaries
        """
        self.logger.info("Processing CSC annotations...")

        # Group by unique conversation (global_context_name + global_speaker)
        grouped_df = df.groupby(['global_context_name', 'global_speaker'])
        
        result = []
        for (context_name, speaker), fields in grouped_df:
            unique_id = f"{context_name}_{speaker}"

            sarcasm_scores = []
            sarcasm_labels = []

            row = fields.iloc[0]
            context_type = row['context_type']
            context_text = row['context_text']
            response_text = row['response_text']
            interlocutor_relationship = row['interlocutor_relationship']
            sarcasm_score_by_speaker = self.normalize_csc_score(row['sarcasm_score_by_speaker'])
            sarcasm_scores.append(sarcasm_score_by_speaker)
            sarcasm_label_by_speaker = int(sarcasm_score_by_speaker >= 0.5) if sarcasm_score_by_speaker is not None else None
            sarcasm_labels.append(sarcasm_label_by_speaker)

            # Build speaker annotation
            speaker_annotation = {
                'annotator_id': str(speaker),
                'irony_appropriateness_label': None,
                'irony_label': None,  # CSC doesn't have irony labels
                'irony_score': None,
                'sarcasm_appropriateness_label': sarcasm_label_by_speaker, # working hypothesis
                'sarcasm_label': sarcasm_label_by_speaker,
                'sarcasm_score': float(sarcasm_score_by_speaker) if pd.notna(sarcasm_score_by_speaker) else None,
                'age': None,  # Not available in CSC
                'gender': None,
                'ethnicity_simplified': None,
                'country_of_birth': None,
                'country_of_residence': None,
                'nationality': None,
                'student_status': None,
                'employment_status': None,
                'generation': None,
                'generation_aggregated': None,
                'employed': None,
                'interlocutor_relationship': interlocutor_relationship
            }

            # get all the df rows that correspond to the current context_name and global_speaker
            observers_df = df[(df['global_context_name'] == context_name) & (df['global_speaker'] == speaker)]
            
            # Build observer annotations and Calculate aggregate sarcasm scores
            observers_annotations = []
            if len(observers_df) > 0:
                for _, observer_row in observers_df.iterrows():
                    global_evaluator = observer_row['global_evaluator']
                    sarcasm_score_by_evaluator = self.normalize_csc_score(observer_row['sarcasm_score_by_evaluator'])
                    sarcasm_scores.append(sarcasm_score_by_evaluator)
                    sarcasm_label_by_evaluator = int(sarcasm_score_by_evaluator >= 0.5) if sarcasm_score_by_evaluator is not None else None
                    sarcasm_labels.append(sarcasm_label_by_evaluator)

                    observer = {
                        'annotator_id': str(global_evaluator),
                        'irony_appropriateness_label': None,
                        'irony_label': None,
                        'irony_score': None,
                        'sarcasm_appropriateness_label': sarcasm_label_by_evaluator, # working hypothesis
                        'sarcasm_label': sarcasm_label_by_evaluator,
                        'sarcasm_score': float(sarcasm_score_by_evaluator) if pd.notna(sarcasm_score_by_evaluator) else None,
                        'age': None,
                        'gender': None,
                        'ethnicity_simplified': None,
                        'country_of_birth': None,
                        'country_of_residence': None,
                        'nationality': None,
                        'student_status': None,
                        'employment_status': None,
                        'generation': None,
                        'generation_aggregated': None,
                        'employed': None,
                    }
                    observers_annotations.append(observer)

            majority_sarcasm_label = self.compute_majority_vote(pd.Series([label for label in sarcasm_labels if label is not None]))
            average_sarcasm_score = np.mean([score for score in sarcasm_scores if score is not None]) if len(sarcasm_scores) > 0 else None
            
            # Create structured entry
            entry = {
                'source': 'fictionary',  # CSC is from fictional scenarios
                'context_type': context_type,
                'context': context_text,
                'post': None,  # CSC doesn't have post
                'reply': response_text,
                'language': 'en',  # CSC is English
                'key_1': str(context_name),
                'key_2': str(speaker),
                'dataset_name': 'CSC',
                'unique_id': unique_id,
                'annotations': {
                    'irony_appropriateness_label': None,
                    'aggregate_irony_label': None,
                    'average_irony_score': None,
                    'sarcasm_appropriateness_label': majority_sarcasm_label, # working hypothesis
                    'aggregate_sarcasm_label': majority_sarcasm_label, # computed by majority vote
                    'average_sarcasm_score': float(average_sarcasm_score) if average_sarcasm_score is not None else None,
                    'speaker_annotation': speaker_annotation,
                    'observers_annotations': observers_annotations
                }
            }
            
            result.append(entry)
        
        self.logger.info(f"Processed {len(result)} unique conversations from CSC")
        return result
    

    def normalize_csc_score(self, score: float | None) -> float | None:
        '''
        min-max normalization of CSC sarcasm scores from 1–6 to 0–1
        '''
        if score is None or pd.isna(score):
            return None
        return (score - 1) / 5  # min-max normalization from 1–6 to 0–1
