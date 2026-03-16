import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random

# Definieren Pfade
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

# Kurze Hilfsfunktion, um JSON-Files einzulesen
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

# Funktion, die die Reviews in Chunks aufteilt, damit sie in das Modell passen. Dabei wird der Präfix mit den Metadaten an den Anfang jedes Chunks gesetzt.
def build_chunks_for_reviews(reviews, tokenizer, max_len=512):
    chunk_texts = []
    chunk_labels = []
    # spezielle Tokens, die das Modell benötigt, um den Anfang und das Ende eines Textes zu markieren, werden berücksichtigt.
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    for rev in reviews:
        review_text = rev.get('review', '')
        m = rev.get('metadata', {})
        # Präfix wird definiert.
        prefix = (
            f"Date: {rev.get('date','')}. "
            f"AverageUserScore: {m.get('averageUserScore','')}. "
            f"GamesReviewed: {m.get('games','')}. "
            f"PositiveReviews: {m.get('scoreCounts',{}).get('positive','')}. "
            f"NeutralReviews: {m.get('scoreCounts',{}).get('neutral','')}. "
            f"NegativeReviews: {m.get('scoreCounts',{}).get('negative','')}"
        )

        # Tokenisierung des Präfixes und des Review-Texts, um die Anzahl der Tokens zu berechnen.
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        review_ids = tokenizer(review_text, add_special_tokens=False)['input_ids']

        # Berechnung der Anzahl der Tokens, die für den Review-Text in jedem Chunk übrig bleiben, nachdem der Präfix und die speziellen Tokens berücksichtigt wurden.
        chunk_body_size = max_len - len(prefix_ids) - special_tokens

        # Aufteilung des Review-Texts in Chunks, die in das Modell passen. -> Code von ChatGPT
        for start in range(0, len(review_ids), chunk_body_size):
            body_slice = review_ids[start:start + chunk_body_size]
            body_text = tokenizer.decode(body_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Hinzufügen des Präfixes zu jedem Chunk.
            chunk_texts.append(prefix + " " + body_text)
            # Die Labels werden in das richtige Format mit 5 Kategorien gebracht.
            chunk_labels.append(score_to_class(rev['rating']))

    return chunk_texts, chunk_labels

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
    json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
    model_out = os.path.join(ROOT, 'model_with_metadata')
    test_out = os.path.join(DATA_DIR, 'test/test_with_metadata.json')

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

    # Der Tokenizer wird geladen.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    # Die Reviews werden in Chunks aufgeteilt, damit sie in das Modell passen. Dabei wird der Präfix mit den Metadaten an den Anfang jedes Chunks gesetzt.
    train_chunk_texts, train_labels = build_chunks_for_reviews(train_reviews, tokenizer, max_len=512)
    eval_chunk_texts, eval_labels = build_chunks_for_reviews(test_reviews, tokenizer, max_len=512)

    # Die Texte werden in Tokens umgewandelt.
    # Diesmal ohne return_overflowing_tokens, da die Chunks bereits in der richtigen Länge zuvor manuell erstellt wurden.
    train_encodings = tokenizer(train_chunk_texts, truncation=True, max_length=512, padding=True)
    eval_encodings = tokenizer(eval_chunk_texts, truncation=True, max_length=512, padding=True)

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
    eval_dataset = SimpleDataset(eval_encodings, eval_labels)

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
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
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