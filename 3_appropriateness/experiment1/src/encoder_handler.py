from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import csv
import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    set_seed
)
from datasets import Dataset
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
from evaluate import load

from src.config import ENCODER_SETTINGS, RANDOM_SEED

class EncoderHandler:
    """
    Handler per modelli encoder che permette training e inferenza
    sia su single-turn (context) che multi-turn (conversation).
    """
    
    INPUT_TYPES = [
        'context',
        'conversation',
    ]
    
    def __init__(self, model_name: str, dataset: Dict[str, Any]):
        """
        Inizializza l'encoder handler.
        
        Args:
            model_name: nome del modello HuggingFace (es. "FacebookAI/xlm-roberta-base")
            dataset: dizionario contenente train, val, test splits e nome dataset
        """
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset['name']
        self.train_split = dataset['train']
        self.val_split = dataset['val']
        self.test_split = dataset['test']
        
        # Carica parametri dal config
        self.training_params = ENCODER_SETTINGS['training_params']
        self.output_dir = ENCODER_SETTINGS['output_dir']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, input_type: str, force_retrain: bool = False) -> str:
        """
        Esegue il fine-tuning del modello encoder.
        
        Args:
            input_type: tipo di input ('context' per single-turn, 'conversation' per multi-turn)
            force_retrain: se True, elimina il modello esistente e riallena da zero
            
        Returns:
            str: percorso del modello fine-tunato
        """
        if input_type not in self.INPUT_TYPES:
            raise ValueError(f"input_type must be one of {self.INPUT_TYPES}, got {input_type}")
        
        print(f"\n{'='*60}")
        print(f"Starting fine-tuning for {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input type: {input_type}")
        print(f"{'='*60}\n")
        
        # Configurazione path
        clean_model_name = self.model_name.replace('/', '_')
        model_output_dir = self.output_dir / clean_model_name / self.dataset_name / input_type
        
        # Verifica se il modello esiste già
        if model_output_dir.exists() and not force_retrain:
            print(f"✓ Model already trained at: {model_output_dir}")
            print("Use force_retrain=True to retrain from scratch")
            return str(model_output_dir)
        
        # Se force_retrain, elimina il modello esistente
        if force_retrain and model_output_dir.exists():
            import shutil
            shutil.rmtree(model_output_dir)
            print(f"✓ Removed existing model at {model_output_dir}")
        
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salva la configurazione degli hyperparametri
        config_file = model_output_dir / "training_config.json"
        with open(config_file, "w", encoding="utf-8") as fo:
            json.dump({
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "input_type": input_type,
                "num_train_epochs": self.training_params['num_train_epochs'],
                "batch_size": self.training_params['batch_size'],
                "learning_rate": self.training_params['learning_rate'],
                "weight_decay": self.training_params['weight_decay'],
                "warmup_ratio": self.training_params['warmup_ratio'],
                "max_length": self.training_params['max_length'],
                "gamma": self.training_params['focal_loss']['gamma'],
                "alpha": self.training_params['focal_loss']['alpha'],
                "seed": RANDOM_SEED
            }, fo, indent=4)
        
        # Carica modello e tokenizer
        self._load_model()
        
        # Prepara i dataset
        set_seed(RANDOM_SEED)
        train_dataset = self._prepare_dataset(self.train_split, input_type)
        val_dataset = self._prepare_dataset(self.val_split, input_type)
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Mostra distribuzione delle classi
        train_labels = np.array(train_dataset["labels"])
        val_labels = np.array(val_dataset["labels"])
        print(f"\nTrain set distribution:")
        print(f"  Class 0 (Not Appropriate): {(train_labels == 0).sum()} ({(train_labels == 0).sum() / len(train_labels) * 100:.1f}%)")
        print(f"  Class 1 (Appropriate): {(train_labels == 1).sum()} ({(train_labels == 1).sum() / len(train_labels) * 100:.1f}%)")
        print(f"Val set distribution:")
        print(f"  Class 0 (Not Appropriate): {(val_labels == 0).sum()} ({(val_labels == 0).sum() / len(val_labels) * 100:.1f}%)")
        print(f"  Class 1 (Appropriate): {(val_labels == 1).sum()} ({(val_labels == 1).sum() / len(val_labels) * 100:.1f}%)")
        
        # compute class weight (in case labels are unbalanced)        
        try:
            labels_array = np.array(train_dataset["labels"]).astype(int)
            class_weights = compute_class_weight(
                "balanced",
                classes=np.unique(labels_array),
                y=labels_array
            ).astype("float32")
            print(f"Class weights: {class_weights}")
        except Exception as e:
            print(f"Unable to balance classes: {e}")
            class_weights = np.array([1.0, 1.0]).astype("float32")
        
        # Training arguments
        training_args = TrainingArguments(
            seed=RANDOM_SEED,
            output_dir=str(model_output_dir),
            num_train_epochs=self.training_params['num_train_epochs'],
            learning_rate=self.training_params['learning_rate'],
            per_device_train_batch_size=self.training_params['batch_size'],
            per_device_eval_batch_size=self.training_params['batch_size'],
            save_strategy=self.training_params['training_args']['save_strategy'],
            eval_strategy=self.training_params['training_args']['eval_strategy'],
            logging_strategy=self.training_params['training_args']['logging_strategy'],
            logging_steps=self.training_params['logging_steps'],
            overwrite_output_dir=self.training_params['training_args']['overwrite_output_dir'],
            report_to=self.training_params['training_args']['report_to'],
            load_best_model_at_end=self.training_params['training_args']['load_best_model_at_end'],
            metric_for_best_model=self.training_params['training_args']['metric_for_best_model'],
            greater_is_better=self.training_params['training_args']['greater_is_better'],
            weight_decay=self.training_params['weight_decay'],
            warmup_ratio=self.training_params['warmup_ratio']
        )
        
        # Trainer con Focal Loss
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        trainer.set_class_weights(class_weights)
        trainer.set_focal_loss_params(
            gamma=self.training_params['focal_loss']['gamma'], 
            alpha=self.training_params['focal_loss']['alpha']
        )
        
        # Training
        print(f"\nStarting training on device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        trainer.train()
        
        # Salva il modello finale
        final_model_path = model_output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        print(f"\n✓ Model saved to: {final_model_path}")
        
        # Salva le metriche di training
        metrics_file = model_output_dir / "training_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(trainer.state.log_history, f, indent=4)
        print(f"✓ Training metrics saved to: {metrics_file}")
        
        # Cleanup
        self._destroy_model()
        
        print(f"\n{'='*60}")
        print(f"Fine-tuning completed for {self.model_name}")
        print(f"{'='*60}\n")
        
        return str(final_model_path)
    
    def predict(self, input_type: str, fine_tuned_model_path: Optional[str] = None):
        """
        Esegue predizioni sul test set.
        
        Args:
            input_type: tipo di input ('context' o 'conversation')
            fine_tuned_model_path: percorso del modello fine-tunato (opzionale, auto-detected)
        """
        if input_type not in self.INPUT_TYPES:
            raise ValueError(f"input_type must be one of {self.INPUT_TYPES}, got {input_type}")
        
        # Auto-detect model path se non specificato
        if fine_tuned_model_path is None:
            clean_model_name = self.model_name.replace('/', '_')
            fine_tuned_model_path = str(
                self.output_dir / clean_model_name / self.dataset_name / input_type / "final_model"
            )
        
        print(f"\n{'='*60}")
        print(f"Running predictions for {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Input type: {input_type}")
        print(f"Model path: {fine_tuned_model_path}")
        print(f"{'='*60}\n")
        
        # Carica il modello fine-tunato
        self._load_fine_tuned_model(fine_tuned_model_path)
        
        # Prepara il test dataset
        test_dataset, test_ids = self._prepare_dataset(self.test_split, input_type, return_ids=True)
        
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Crea trainer per predizione
        trainer = Trainer(
            model=self.model,
            compute_metrics=self._compute_metrics
        )
        
        # Esegui predizioni
        print("\nGenerating predictions...")
        predictions = trainer.predict(test_dataset)
        
        # Salva risultati
        clean_model_name = self.model_name.replace('/', '_')
        prediction_dir = self.output_dir / clean_model_name / self.dataset_name / input_type / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = prediction_dir / f"fine_tuning_{clean_model_name}_{self.dataset_name}_{input_type}.csv"
        
        with open(output_file, "w", newline='', encoding='utf-8') as fo:
            writer = csv.DictWriter(fo, fieldnames=["unique_id", "model_answer", "label"])
            writer.writeheader()
            
            pred_labels = np.argmax(predictions.predictions, axis=-1)
            true_labels = predictions.label_ids
            
            for unique_id, pred_label, true_label in zip(test_ids, pred_labels, true_labels):
                writer.writerow({
                    "unique_id": unique_id,
                    "model_answer": int(pred_label),
                    "label": int(true_label)
                })
        
        # Stampa metriche
        print("\n" + "="*60)
        print("Test Set Results:")
        print("="*60)
        metrics = predictions.metrics
        for key, value in metrics.items():
            if key.startswith('test_'):
                print(f"{key}: {value:.4f}")
        
        # Classification report dettagliato
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels, 
                                   target_names=['Not Appropriate', 'Appropriate']))
        
        print(f"\n✓ Predictions saved to: {output_file}")
        print("="*60 + "\n")
        
        # Cleanup
        self._destroy_model()
    
    def _prepare_dataset(self, split: List[Dict], input_type: str, return_ids: bool = False):
        """
        Prepara il dataset per training o inferenza.
        
        Args:
            split: lista di entry del dataset
            input_type: 'context' o 'conversation'
            return_ids: se True, ritorna anche gli unique_id
        """
        ids, texts, labels = [], [], []
        
        for entry in split:
            ids.append(entry['unique_id'])
            
            # Prepara il testo in base all'input_type
            if input_type == 'context':
                if self.dataset_name == 'MultiPICo':
                    text = f"{entry['post']}"
                else:
                    text = f"{entry['context']}"
            elif input_type == 'conversation':
                # Multi-turn: usa l'intera conversazione
                dialogue = entry['dialogue']
                if 'turns' in dialogue:
                    # Copia i turni per non modificare l'originale
                    turns_copy = [dict(t) for t in dialogue['turns']]
                    # Rimuovi SOLO l'u2 dall'ULTIMO turno (la risposta target da predire)
                    if turns_copy and 'u2' in turns_copy[-1]:
                        del turns_copy[-1]['u2']
                    
                    # Costruisci il testo del dialogo
                    turns_str = ""
                    for turn in turns_copy:
                        # Ora alcuni turni avranno sia u1 che u2, l'ultimo solo u1
                        if 'u1' in turn:
                            turns_str += f"u1: {turn['u1']}\n"
                        if 'u2' in turn:
                            turns_str += f"u2: {turn['u2']}\n"
                    text = turns_str.strip()
                else:
                    raise ValueError("Dialogue format not recognized. Expected 'turns' key.")
            else:
                raise ValueError(f"input_type must be one of {self.INPUT_TYPES}, got {input_type}")
            
            texts.append(text)
            labels.append(entry['label'])
        
        # Crea dataset
        df = pd.DataFrame({"text": texts, "labels": labels})
        dt = Dataset.from_pandas(df)
        
        # Tokenizza
        tokenized_dataset = dt.map(
            lambda x: self._tokenize(x), 
            remove_columns=['text'],
            batched=True
        )
        tokenized_dataset = tokenized_dataset.with_format("torch")
        
        if return_ids:
            return tokenized_dataset, ids
        return tokenized_dataset
    
    def _tokenize(self, examples):
        """Tokenizza il testo."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.training_params['max_length'],
        )
    
    @staticmethod
    def _compute_metrics(eval_pred):
        """Calcola metriche di valutazione focalizzate sulla classe positiva (label=1)."""
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        # Carica le metriche da Hugging Face Evaluate
        accuracy_metric = load("accuracy")
        precision_metric = load("precision")
        recall_metric = load("recall")
        f1_metric = load("f1")
        
        # Calcola metriche per la classe positiva (pos_label=1, average="binary")
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=preds, references=labels, pos_label=1, average="binary")["precision"]
        recall = recall_metric.compute(predictions=preds, references=labels, pos_label=1, average="binary")["recall"]
        f1 = f1_metric.compute(predictions=preds, references=labels, pos_label=1, average="binary")["f1"]
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def _load_model(self):
        """Carica il modello base e il tokenizer."""
        print(f"\nLoading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        print(f"✓ Loaded tokenizer for {self.model_name}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2, 
            trust_remote_code=True
        )
        self.model.to(self.device)
        print(f"✓ Loaded model for sequence classification with 2 labels")
        print(f"✓ Model moved to device: {self.device}")
    
    def _load_fine_tuned_model(self, model_path: str):
        """Carica un modello fine-tunato."""
        print(f"\nLoading fine-tuned model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print(f"✓ Loaded tokenizer")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.to(self.device)
        print(f"✓ Loaded fine-tuned model")
        print(f"✓ Model moved to device: {self.device}")
    
    def _destroy_model(self):
        """Libera la memoria del modello."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✓ Model memory cleared")


class CustomTrainer(Trainer):
    """
    Custom Trainer class con Focal Loss per gestire classi sbilanciate.
    Focal Loss: https://arxiv.org/abs/1708.02002
    """
    
    def set_class_weights(self, class_weights):
        """Imposta i pesi delle classi."""
        self.class_weights = class_weights
    
    def set_focal_loss_params(self, gamma: float, alpha: float):
        """Imposta i parametri della Focal Loss."""
        self.gamma = gamma
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Calcola la Focal Loss invece della standard Cross Entropy.
        Focal Loss: https://arxiv.org/abs/1708.02002
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Fix per DataParallel
        real_model = model.module if hasattr(model, "module") else model
        num_labels = real_model.config.num_labels
        
        # Cross Entropy Loss con class weights
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=logits.device)
        )
        
        BCEloss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        # Focal Loss
        pt = torch.exp(-BCEloss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCEloss
        
        return (loss, outputs) if return_outputs else loss
