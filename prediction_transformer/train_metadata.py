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

def build_chunks_for_reviews(reviews, tokenizer, max_len=512):
    chunk_texts = []
    chunk_labels = []
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)


    for rev in reviews:
        review_text = rev.get('review', '')
        m = rev.get('metadata', {})
        prefix = (
            f"Date: {rev.get('date','')}. "
            f"AverageUserScore: {m.get('averageUserScore','')}. "
            f"GamesReviewed: {m.get('games','')}. "
            f"PositiveReviews: {m.get('scoreCounts',{}).get('positive','')}. "
            f"NeutralReviews: {m.get('scoreCounts',{}).get('neutral','')}. "
            f"NegativeReviews: {m.get('scoreCounts',{}).get('negative','')}"
        )

        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        review_ids = tokenizer(review_text, add_special_tokens=False)['input_ids']

        chunk_body_size = max_len - len(prefix_ids) - special_tokens

        for start in range(0, len(review_ids), chunk_body_size):
            body_slice = review_ids[start:start + chunk_body_size]
            body_text = tokenizer.decode(body_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunk_texts.append(prefix + " " + body_text)
            chunk_labels.append(score_to_class(rev['rating']))

    return chunk_texts, chunk_labels

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

def train_on_file():
    json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
    model_out = os.path.join(MODELS_DIR, 'model_with_metadata')
    test_out = os.path.join(DATA_DIR, 'test/test_with_metadata.json')

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    train_chunk_texts, train_labels = build_chunks_for_reviews(train_reviews, tokenizer, max_len=512)
    eval_chunk_texts, eval_labels = build_chunks_for_reviews(test_reviews, tokenizer, max_len=512)

    train_encodings = tokenizer(train_chunk_texts, truncation=True, max_length=512, padding=True)
    eval_encodings = tokenizer(eval_chunk_texts, truncation=True, max_length=512, padding=True)

    class_counts = [train_labels.count(i) for i in range(5)]
    class_counts = [c if c > 0 else 1 for c in class_counts]
    class_weights = [len(train_labels) / c for c in class_counts]
    class_weights = torch.tensor(class_weights).to(DEVICE)

    train_dataset = SimpleDataset(train_encodings, train_labels)
    eval_dataset = SimpleDataset(eval_encodings, eval_labels)

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
    train_on_file()