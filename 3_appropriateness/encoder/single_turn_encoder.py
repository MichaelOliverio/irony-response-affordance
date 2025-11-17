import sys
import os

import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, set_seed, TrainingArguments
from datasets import Dataset
import torch
from sklearn.utils import compute_class_weight
import numpy as np
from tqdm import tqdm
import json
import datasets
from sklearn.model_selection import train_test_split
from evaluate import load
import numpy as np

SEED = 42
NUM_TRAIN_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1 # aumenta gradualmente il learning rate
GAMMA = 5.0
ALPHA = 0.2

class Encoder():
    def __init__(self, model_identifier, dataset):
        self.model_id = model_identifier
        self.training_split = dataset["training_set"]
        self.validation_split = dataset["validation_set"]
        self.test_split = dataset["test_set"]
        self.dataset_name = dataset["dataset_name"] 
        self.dataset_language = dataset["dataset_language"]
        self.prediction_dir = f"./models/{model_identifier}_fine_tuned/predictions"
        self.output_dir = f"./models/{model_identifier}_fine_tuned"
        self.seed = SEED

        # salva la configurazione degli hyperparametri
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(f"{self.output_dir}/training_config.json", "w", encoding="utf-8") as fo:
            json.dump({
                "num_train_epochs": NUM_TRAIN_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_ratio": WARMUP_RATIO,
                "gamma": GAMMA,
                "alpha": ALPHA
            }, fo, indent=4)
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, trust_remote_code=True)
        print(f"Loaded tokenizer for model: {model_identifier}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2, trust_remote_code=True)
        print(f"Loaded model for sequence classification: {model_identifier} with 2 labels")


    def train(self):
        set_seed(self.seed)
        data = {
            "train" : self.__generate_data(self.training_split)[0], # tokenized dataset
            "val" : self.__generate_data(self.validation_split)[0] # tokenized dataset
        }   

        # compute class weight (in case labels are unbalanced)        
        try:
            labels_array = np.array(data["train"]["labels"]).astype(int)

            class_weights = compute_class_weight(
                "balanced",
                classes=np.unique(labels_array),
                y=labels_array
            ).astype("float32")
        except Exception as e:
            print("Unable to balance classes")
            class_weights = np.array([1, 1]).astype("float32")

        print("Class weights:", class_weights)


        print('We will use the device:', torch.cuda.get_device_name(0))
        training_args = TrainingArguments(
            seed=self.seed,
            output_dir=self.output_dir,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=LOGGING_STEPS, # ogni quanti step loggare
            overwrite_output_dir=True,
            report_to=None,
            load_best_model_at_end=True,
            weight_decay=WEIGHT_DECAY,   # Standard per BERT/RoBERTa (CONSIGLIATO)
            warmup_ratio=WARMUP_RATIO # aumenta gradualmente il learning rate
        )


        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=data["train"],
            eval_dataset=data["val"],
            compute_metrics=self.__compute_metrics
        )
        trainer.set_class_weights(class_weights)
        trainer.train()
        return trainer
    

    def predict(self, trainer):
        test_data, ids = self.__generate_data(self.test_split)
        predictions = trainer.predict(test_data)

        if not os.path.exists(self.prediction_dir): 
            os.makedirs(self.prediction_dir)  
        with open(f"{self.prediction_dir}/predictions_{self.dataset_name}_{self.dataset_language}.csv", "w", newline='', encoding='utf-8') as fo:
            writer = csv.DictWriter(fo, fieldnames=["unique_id", "predicted_label", "label"])
            writer.writeheader()
            for i, pred in enumerate(predictions.predictions):
                predicted_label = int(np.argmax(pred))
                writer.writerow({
                    "unique_id": ids[i],
                    "predicted_label": predicted_label,
                    "label": int(predictions.label_ids[i])
                })


    def __generate_data(self, split):
        ids, texts, labels = [], [], []

        for _, row in split.iterrows():
            ids.append(row['unique_id'])
            texts.append(row['text'])
            labels.append(row['label'])

        df = pd.DataFrame({"text":texts, "labels":labels})
        dt = Dataset.from_pandas(df)
        tokenized_dataset = dt.map(lambda x: self.__tokenize(x, self.tokenizer), remove_columns=['text'])
        tokenized_dataset = tokenized_dataset.with_format("torch")
        return tokenized_dataset, ids
    
    @staticmethod
    def __compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        # Carica le metriche da Hugging Face Evaluate
        accuracy_metric = load("accuracy")
        precision_metric = load("precision")
        recall_metric = load("recall")
        f1_metric = load("f1")
        
        # Calcola tutte le metriche
        accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        precision = precision_metric.compute(predictions=preds, references=labels, average="weighted")["precision"]
        recall = recall_metric.compute(predictions=preds, references=labels, average="weighted")["recall"]
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


    def __tokenize(self, x, tokenizer):
        return tokenizer(
            x["text"], 
            padding="max_length", #config.padding,
            truncation=True, #config.truncation,
            max_length=512, #config.max_length,
        )

class CustomTrainer(Trainer):
    """ Custom Trainer class to implement a custom loss function
    """

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Focal Loss: https://arxiv.org/abs/1708.02002
        """
        gamma = GAMMA
        alpha = ALPHA
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Focal Loss Perseval
        # -----------------------------
        # FIX: gestiamo DataParallel
        # -----------------------------
        real_model = model.module if hasattr(model, "module") else model
        num_labels = real_model.config.num_labels
        # -----------------------------

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to(logits.device)

        BCEloss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        pt = torch.exp(-BCEloss)
        loss = alpha * (1 - pt) ** gamma * BCEloss

        return (loss, outputs) if return_outputs else loss


def load_dataset(file_path, language='en', dataset_name='MultiPICo', limit=None):
    """
    Carica il dataset dal unified corpus, mantenendo gli unique_id anche negli split.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            unified_corpus = json.load(f)
        print(f"Loaded unified corpus with {len(unified_corpus)} entries")
        
        df = pd.DataFrame(unified_corpus)
        
        # Filtra per lingua e dataset
        df_filtered = df[(df['language'] == language) & (df['dataset_name'] == dataset_name)]
        print(f"Filtered {dataset_name} {language.upper()} entries: {len(df_filtered)}")
        
        # Limita il numero di entries se necessario
        if limit is not None:
            df_filtered = df_filtered.head(limit)
            print(f"Using {len(df_filtered)} entries (limited)")

        # Seleziona solo le colonne utili
        df_filtered = df_filtered[['unique_id', 'post', 'annotations']].copy()
        df_filtered = df_filtered.rename(columns={'post': 'text'})
        df_filtered['label'] = df_filtered['annotations'].apply(
            lambda x: x.get('irony_appropriateness_label', None)
        )
        df_filtered = df_filtered.drop(columns=['annotations'])

        print(f"Entries with label 1: {len(df_filtered[df_filtered['label'] == 1])}")
        print(f"Entries with label 0: {len(df_filtered[df_filtered['label'] == 0])}")

        # Includiamo anche gli ID nello split
        train_df, temp_df = train_test_split(
            df_filtered,
            test_size=0.3,
            random_state=42,
            stratify=df_filtered['label']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['label']
        )

        # Mostra distribuzioni
        for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            print(f'{name} len: {len(split_df)} | label 1: {len(split_df[split_df["label"]==1])} | label 0: {len(split_df[split_df["label"]==0])}')

        dataset_splits = {
            'dataset_name': dataset_name,
            'dataset_language': language,
            'training_set': train_df.reset_index(drop=True),
            'validation_set': val_df.reset_index(drop=True),
            'test_set': test_df.reset_index(drop=True)
        }

        return dataset_splits
        
    except Exception as e:
        print(f"Error loading unified corpus: {e}")
        raise e

def main():
    data_path = "../../1_datasets_merge/output/unified_corpus.json"

    # Carica e filtra il dataset
    dataset_splits = load_dataset(
        file_path=data_path,
        language='en',
        dataset_name='MultiPICo',
        limit=None  # None per usare tutti i dati, oppure un numero specifico per test
    )

    models = {
        #"eurobert": "EuroBERT/EuroBERT-210m",
        "xlm-roberta-base": "FacebookAI/xlm-roberta-base"
    }

    for model_name, model_identifier in models.items():
        print(f"Training and evaluating model: {model_name} ({model_identifier})")
        model = Encoder(model_identifier, dataset_splits)
        trainer = model.train()
        model.predict(trainer)

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()