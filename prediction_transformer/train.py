import os
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.model_selection import train_test_split

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'prediction_transformer/models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_BASE = 'xlm-roberta-base'

def load_json(path):
    with open(path, encoding='utf-8') as f:
        j = json.load(f)
        return j

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare(items):
    children = [
        review
        for reviews in items.values()
        if isinstance(reviews, list)
        for review in reviews
    ]

    return children

def prepareData(reviews, metadata):
    if metadata:
        m = reviews["metadata"]
        return f"""
        Date: {reviews['date']}
        AverageUserScore: {m['averageUserScore']}
        GamesReviewed: {m['games']}
        PositiveReviews: {m['scoreCounts']['positive']}
        NeutralReviews: {m['scoreCounts']['neutral']}
        NegativeReviews: {m['scoreCounts']['negative']}
        Review: {reviews['review']}
        """
    else:
        return f"""
        Date: {reviews['date']}
        Review: {reviews['review']}
        """

def train_on_file(metadata = False):
    if metadata:
        json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
        model_out = os.path.join(MODELS_DIR, 'model_with_metadata')
        test_out = os.path.join(DATA_DIR, 'test_with_metadata.json')
        os.makedirs(model_out, exist_ok=True)
    else:
        json_path = os.path.join(DATA_DIR, 'all_without_metadata.json')
        model_out = os.path.join(MODELS_DIR, 'model_without_metadata')
        test_out = os.path.join(DATA_DIR, 'test_without_metadata.json')
        os.makedirs(model_out, exist_ok=True)

    items = load_json(json_path)
    print(f"Loaded {len(items)} items from {json_path}")
    reviews = prepare(items)
    print(f"Prepared {len(reviews)} game reviews for training")

    train_reviews, test_reviews = train_test_split(reviews, test_size=0.25, random_state=42)
    print(f"Split into {len(train_reviews)} train and {len(test_reviews)} test reviews")

    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_reviews, f, indent=2)

    print(f"Saved test data to {test_out}")

    train_texts = [prepareData(review, metadata) for review in train_reviews]
    train_labels = [float(review['rating']) for review in train_reviews]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    train_dataset = SimpleDataset(encodings, train_labels)

    # eval dataset
    eval_texts = [prepareData(review, metadata) for review in test_reviews]
    eval_labels = [float(review['rating']) for review in test_reviews]
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=512)
    eval_dataset = SimpleDataset(eval_encodings, eval_labels)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE, ignore_mismatched_sizes=True, num_labels=1)

    training_args = TrainingArguments(
        output_dir=model_out,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_strategy='epoch',
        eval_strategy='epoch',
        learning_rate=2e-5,
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    trainer.save_model(model_out)
    print(f"Saved trained model to {model_out}")

if __name__ == '__main__':
    print(torch.cuda.is_available())
    # train_on_file(True)
    train_on_file(False)
