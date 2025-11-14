import pandas as pd
from typing import Optional, List, Dict
from .config import ACCEPTED_INPUT_TYPES, ACCEPTED_OUTPUT_TYPES, DATA_AUGMENTATION_DIR
from .dialogue_generator import DialogueGenerator

class DialogueDataAugmentor:
    @staticmethod
    def augment_data(
        df: pd.DataFrame, 
        input_type: str = "single-turn", 
        output_type: str = "multi-turn",
    ) -> Dict[str, str]:
        """
        Esegue la data augmentation sul DataFrame fornito.
        
        Args:
            df: DataFrame contenente i dati da aumentare
            input_type: Tipo di input ("single-turn" o "contextual")
            output_type: Tipo di output desiderato ("multi-turn")
            models: Lista di modelli da usare (default: DialogueGenerator.DEFAULT_MODELS)
            batch_size: Dimensione del batch (default: 1)
            max_retries: Numero massimo di retry per inferenze fallite (default: 3)
            temperature: Temperatura per il sampling (default: 0.1)
            output_dir: Directory dove salvare i risultati (default: DATA_AUGMENTATION_DIR)
        
        Returns:
            Dizionario {model_name: output_file_path}
        """
        if input_type not in ACCEPTED_INPUT_TYPES:
            raise ValueError(f"Input type non valido. Accettati: {ACCEPTED_INPUT_TYPES}")
        if output_type not in ACCEPTED_OUTPUT_TYPES:
            raise ValueError(f"Output type non valido. Accettati: {ACCEPTED_OUTPUT_TYPES}")

        return DialogueDataAugmentor._compute_data_augmentation(
            df, input_type, output_type
        )
    
    @staticmethod
    def _compute_data_augmentation(
        df: pd.DataFrame, 
        input_type: str, 
        output_type: str,
    ) -> Dict[str, str]:
        """Esegue la data augmentation in base ai tipi di input/output."""
        if input_type == "single-turn" and output_type == "multi-turn":
            return DialogueDataAugmentor._augment_single_to_multi(df)
        elif input_type == "contextual" and output_type == "multi-turn":
            return DialogueDataAugmentor._augment_contextual_to_multi(df)
        else:
            raise NotImplementedError(
                f"Data augmentation non implementata per input_type={input_type} e output_type={output_type}"
            )

    @staticmethod
    def _augment_single_to_multi(
        df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Augmentation da single-turn a multi-turn.
        
        Usa DialogueGenerator per generare dialoghi multi-turn
        da coppie post-reply singole.
        """
        print("\n=== Data Augmentation: single-turn → multi-turn ===")
        
        return DialogueGenerator.generate_dialogues_for_dataset(df)

    @staticmethod
    def _augment_contextual_to_multi(
        df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Augmentation da contextual a multi-turn.
        
        TODO: Implementare logica specifica per dati contestuali.
        Per ora usa la stessa logica di single-turn.
        """
        print("\n=== Data Augmentation: contextual → multi-turn ===")
        print("⚠️  Usando logica single-turn (da implementare logica specifica)")

        return DialogueGenerator.generate_dialogues_for_dataset(df)