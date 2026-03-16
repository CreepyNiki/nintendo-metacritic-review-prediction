import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random

# Pfade werden definiert.
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(ROOT, exist_ok=True)

# Modell, dass gefinetuned wird.
MODEL_BASE = 'roberta-base'

# mit GPU trainieren lassen wenn möglich.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Alle möglichen Zufallszahlen auf 42 setzen, damit die Ergebnisse reproduzierbar sind.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Kurze Hilfsfunktion, um JSON-Files einzulesen.
def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# Eigener Trainer wird definiert, um die Klassen-Gewichte in die Verlustfunktion einzubauen. Dabei wird die computer_loss Funktion des Ursprungstrainers überschrieben.
# Implementation des Trainers und der Methode von Copilot.
# Idee von https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# Inspiriert von: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Hilfsfunktion, die die Reviews aus den JSON-Dateien extrahiert. Die Reviews werden in einer Liste zurückgegeben.
def prepare(items):
    return [
        review
        for reviews in items.values()
        if isinstance(reviews, list)
        for review in reviews
    ]

# Funktion, die die Review-Text und das Datum in einem String zusammenfasst. Das Datum wird hierbei vor den Reviewtext geschrieben.
def prepareData(review):
        return f"""
        Date: {review['date']}
        Review: {review['review']}
        """

# Funktion, die den numerischen Score in eine von 5 Klassen umwandelt. Dabei werden die Scores in folgende Klassen eingeteilt:
# 0: Score 0-1 -> sehr schlechte Bewertung
# 1: Score 2-3 -> schlechte Bewertung
# 2: Score 4-6 -> durchschnittliche Bewertung
# 3: Score 7-8 -> gute Bewertung
# 4: Score 9-10 -> sehr gute Bewertung
def score_to_class(score):
    score = int(score)
    if score <= 1:
        return 0
    elif score <= 3:
        return 1
    elif score <= 6:
        return 2
    elif score <= 8:
        return 3
    else:
        return 4

# Hauptmethode, mit der das Basismodell trainiert und gefinetuned wird.
def train_on_file():
    # Pfade werden definiert
    json_path = os.path.join(DATA_DIR, 'all_without_metadata.json')
    model_out = os.path.join(ROOT, 'model_without_metadata')
    test_out = os.path.join(DATA_DIR, 'test/test_without_metadata.json')

    os.makedirs(model_out, exist_ok=True)

    # Daten werden reingeladen und vorbereitet.
    items = load_json(json_path)
    reviews = prepare(items)
    print(f"Loaded {len(items)} items from {json_path}")
    print(f"Prepared {len(reviews)} game reviews for training")

    # Train-Test-Split wird durchgeführt. Trainingsgröße: 80%, Testgröße 20%
    train_reviews, test_reviews = train_test_split(reviews, test_size=0.2, random_state=42)
    print(f"Split into {len(train_reviews)} train and {len(test_reviews)} test reviews")

    # Testdaten werden in einem eigenen JSON-File gespeichert, damit sie später für die Evaluation genutzt werden können.
    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_reviews, f, indent=2)
    print(f"Saved test data to {test_out}")

    # Daten werden in das richtige Format gebracht
    train_texts = [prepareData(r) for r in train_reviews]
    eval_texts = [prepareData(r) for r in test_reviews]

    # Die Tokenizer werden geladen.
    # Dabei wird die maximale Länge auf 512 Tokens gesetzt, damit sie in das Modell passen.
    # Außerdem wird die Option "return_overflowing_tokens" genutzt, damit zu lange Reviews in mehrere Chunks aufgeteilt werden können, die jeweils in das Modell passen.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    # Die Texte werden in Tokens umgewandelt.
    train_encodings = tokenizer(train_texts, truncation=True, max_length=512, padding=True, stride=128, return_overflowing_tokens=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, max_length=512, padding=True, stride=128, return_overflowing_tokens=True)

    # Die "overflow_to_sample_mapping" wird genutzt, um die Labels den Chunks zuordnen zu können. Diese wird nicht mehr benötigt.
    mapping = train_encodings.pop("overflow_to_sample_mapping")
    eval_mapping = eval_encodings.pop("overflow_to_sample_mapping")

    print(f"Training and testing label distribution:")
    # Die Labels werden den Chunks zugeordnet und dabei in das richtige Format mit 5 Kategorien gebracht.
    train_labels = [score_to_class(train_reviews[idx]["rating"]) for idx in mapping]
    test_labels = [score_to_class(test_reviews[idx]["rating"]) for idx in eval_mapping]
    for i in range(5):
        # Die Verteilung der Klassen  wird ausgegeben.
        print(f"  Class {i}: Train={train_labels.count(i)}, Test={test_labels.count(i)}")

    # Die Klassen-Gewichte werden berechnet, während des Trainings die Klassen entsprechend ihrer Häufigkeit gewichtet werden.
    class_counts = [train_labels.count(i) for i in range(5)]
    # Wenn eine Klasse nicht vorhanden ist, wird das Gewicht auf 1 gesetzt, um Division durch Null zu vermeiden.
    class_counts = [c if c > 0 else 1 for c in class_counts]
    # Anzahl der Samples durch die Anzahl der Samples pro Klasse geteilt, um die Gewichte zu berechnen.
    class_weights = [len(train_labels) / c for c in class_counts]
    # Umwanldung in Tensor und hinzufügen zu GPU oder CPU.
    class_weights = torch.tensor(class_weights).to(DEVICE)

    # Datasets werden erstellt.
    train_dataset = SimpleDataset(train_encodings, train_labels)
    eval_dataset = SimpleDataset(eval_encodings, test_labels)

    # Modell wird geladen und GPU oder CPU hinzugefügt.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=5, problem_type="single_label_classification"
    ).to(DEVICE)

    # Hilfsfunktion, die die benötigten Metriken für die Evaluation berechnet.
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
        }

    # Trainingsargumente werden definiert.
    training_args = TrainingArguments(
        output_dir=model_out,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        weight_decay=0.01,
        fp16=True,
        seed=seed,
    )

    # Erstellen des Custom Trainers.
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    # Training und Speichern des Modells.
    trainer.train()
    trainer.save_model(model_out)
    print(f"Saved trained model to {model_out}")

if __name__ == "__main__":
    train_on_file()