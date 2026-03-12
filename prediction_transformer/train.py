import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import random

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'prediction_transformer/models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_BASE = 'roberta-base'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

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

def prepare(items):
    return [
        review
        for reviews in items.values()
        if isinstance(reviews, list)
        for review in reviews
    ]

def prepareData(review, metadata):
    if metadata:
        m = review["metadata"]
        return f"""
        Date: {review['date']}
        AverageUserScore: {m['averageUserScore']}
        GamesReviewed: {m['games']}
        PositiveReviews: {m['scoreCounts']['positive']}
        NeutralReviews: {m['scoreCounts']['neutral']}
        NegativeReviews: {m['scoreCounts']['negative']}
        Review: {review['review']}
        """
    else:
        return f"""
        Date: {review['date']}
        Review: {review['review']}
        """

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

def train_on_file(metadata=False):
    if metadata:
        json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
        model_out = os.path.join(MODELS_DIR, 'model_with_metadata')
        test_out = os.path.join(DATA_DIR, 'test/test_with_metadata.json')
    else:
        json_path = os.path.join(DATA_DIR, 'all_without_metadata.json')
        model_out = os.path.join(MODELS_DIR, 'model_without_metadata')
        test_out = os.path.join(DATA_DIR, 'test/test_without_metadata.json')

    os.makedirs(model_out, exist_ok=True)

    items = load_json(json_path)
    reviews = prepare(items)
    print(f"Loaded {len(items)} items from {json_path}")
    print(f"Prepared {len(reviews)} game reviews for training")

    train_reviews, test_reviews = train_test_split(reviews, test_size=0.2, random_state=42)
    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_reviews, f, indent=2)

    print(f"Saved test data to {test_out}")
    print(f"Split into {len(train_reviews)} train and {len(test_reviews)} test reviews")

    train_texts = [prepareData(r, metadata) for r in train_reviews]
    eval_texts = [prepareData(r, metadata) for r in test_reviews]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    train_encodings = tokenizer(train_texts, truncation=True, max_length=512, padding=True, stride=128, return_overflowing_tokens=True)
    eval_encodings = tokenizer(eval_texts, truncation=True, max_length=512, padding=True, stride=128, return_overflowing_tokens=True)

    mapping = train_encodings.pop("overflow_to_sample_mapping")
    eval_mapping = eval_encodings.pop("overflow_to_sample_mapping")

    print(f"Training and testing label distribution:")
    train_labels = [score_to_class(train_reviews[idx]["rating"]) for idx in mapping]
    test_labels = [score_to_class(test_reviews[idx]["rating"]) for idx in eval_mapping]
    for i in range(5):
        print(f"  Class {i}: Train={train_labels.count(i)}, Test={test_labels.count(i)}")

    class_counts = [train_labels.count(i) for i in range(5)]
    class_counts = [c if c > 0 else 1 for c in class_counts]
    class_weights = [len(train_labels) / c for c in class_counts]
    class_weights = torch.tensor(class_weights).to(DEVICE)

    train_dataset = SimpleDataset(train_encodings, train_labels)
    eval_dataset = SimpleDataset(eval_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE, num_labels=5, problem_type="single_label_classification"
    ).to(DEVICE)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
        }

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

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(model_out)
    print(f"Saved trained model to {model_out}")

if __name__ == "__main__":
    train_on_file(False)
    # train_on_file(True)