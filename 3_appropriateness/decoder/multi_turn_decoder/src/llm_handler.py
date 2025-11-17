"""
Gestione del caricamento e distruzione dei modelli LLM.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Tuple, Optional
from huggingface_hub import login
import os
from dotenv import load_dotenv


class LLMHandler:
    """Classe per gestire il caricamento e la distruzione di modelli LLM."""
    
    _hf_authenticated = False
    
    @staticmethod
    def authenticate_huggingface():
        """Autentica con HuggingFace Hub usando il token da .env"""
        if not LLMHandler._hf_authenticated:
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                login(token=hf_token)
                LLMHandler._hf_authenticated = True
                print("Autenticazione HuggingFace completata")
            else:
                print("Warning: HF_TOKEN non trovato nel file .env")
    
    @staticmethod
    def get_device() -> int:
        """Restituisce il device da usare (GPU o CPU)."""
        return 0 if torch.cuda.is_available() else -1
    
    @staticmethod
    def load_model(model_name: str, dtype=torch.float16):
        """
        Carica un modello LLM e il suo tokenizer.
        
        Args:
            model_name: Nome del modello da caricare (es. 'meta-llama/Llama-3.2-3B-Instruct')
            dtype: Tipo di dati per il modello (default: torch.float16)
        
        Returns:
            Tuple[pipeline, model, tokenizer]
        """
        LLMHandler.authenticate_huggingface()
        
        print(f"Caricamento modello: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda" if torch.cuda.is_available() else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Se manca pad token, usiamo eos come pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Impostiamo padding_side='left' per decoder-only models
        tokenizer.padding_side = "left"

        # Creiamo la pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
        )

        # Assicuriamoci che la pipeline conosca il pad_token_id
        pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

        print(f"Modello caricato: {model_name}")
        print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        
        return pipe, model, tokenizer

    @staticmethod
    def destroy_model(model, tokenizer):
        """
        Distrugge il modello e il tokenizer per liberare memoria.
        
        Args:
            model: Modello da distruggere
            tokenizer: Tokenizer da distruggere
        """
        if model is not None:
            del model
        
        if tokenizer is not None:
            del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Modello distrutto e cache GPU svuotata")
        else:
            print("Modello distrutto")
