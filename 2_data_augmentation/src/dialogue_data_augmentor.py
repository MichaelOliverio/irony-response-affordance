import pandas as pd
from .config import ACCEPTED_INPUT_TYPES, ACCEPTED_OUTPUT_TYPES

class DialogueDataAugmentor:
    @staticmethod
    def augment_data(df, input_type: str = "single-turn", output_type: str = "multi-turn") -> pd.DataFrame:
        """
        Esegue la data augmentation sul DataFrame fornito.
        
        Args:
            df: DataFrame contenente i dati da aumentare
            input_type: Tipo di input ("single-turn" o "multi-turn")
            output_type: Tipo di output desiderato ("single-turn" o "multi-turn")
        
        Returns:
            DataFrame con i dati aumentati
        """

        if input_type not in ACCEPTED_INPUT_TYPES:
            raise ValueError(f"Input type non valido. Accettati: {ACCEPTED_INPUT_TYPES}")
        if output_type not in ACCEPTED_OUTPUT_TYPES:
            raise ValueError(f"Output type non valido. Accettati: {ACCEPTED_OUTPUT_TYPES}")

        augmented_df = DialogueDataAugmentor._compute_data_augmentation(df, input_type, output_type)
        return augmented_df
    
    @staticmethod
    def _compute_data_augmentation(df: pd.DataFrame, input_type: str, output_type: str) -> pd.DataFrame:
        if input_type == "single-turn" and output_type == "multi-turn":
            return DialogueDataAugmentor._augment_single_to_multi(df)
        elif input_type == "contextual" and output_type == "multi-turn":
            return DialogueDataAugmentor._augment_contextual_to_multi(df)
        else:
            raise NotImplementedError(f"Data augmentation non implementata per input_type={input_type} e output_type={output_type}")

    @staticmethod
    def _augment_single_to_multi(df: pd.DataFrame) -> pd.DataFrame:
        # Logica per l'augmentation da single-turn a multi-turn
        return df

    @staticmethod
    def _augment_contextual_to_multi(df: pd.DataFrame) -> pd.DataFrame:
        # Logica per l'augmentation da contextual a multi-turn
        return df