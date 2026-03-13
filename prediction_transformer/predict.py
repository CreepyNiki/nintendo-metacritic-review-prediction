import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from collections import defaultdict

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "prediction_transformer/models/model_without_metadata")
GOLDSTANDARD_PATH = os.path.join(DATA_DIR, "../model_without_metadata")

MODEL_BASE = "roberta-base"

def load_json():
    path = os.path.join(DATA_DIR, "test/test_without_metadata.json")
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def prepareData(review):
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

def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    model_path = MODEL_DIR
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_data = load_json()
    print("Test size:", len(test_data))

    texts = [prepareData(r) for r in test_data]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
        stride=128,
        return_overflowing_tokens=True
    )

    mapping = encodings.pop("overflow_to_sample_mapping")

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        chunk_preds = torch.argmax(logits, dim=1).cpu().numpy()

    if not isinstance(mapping, list):
        try:
            mapping = mapping.tolist()
        except:
            mapping = list(mapping)

    review_predictions = defaultdict(list)
    for chunk_idx, sample_idx in enumerate(mapping):
        pred = int(chunk_preds[chunk_idx])
        review_predictions[sample_idx].append(pred)

    final_preds = []
    for i in range(len(test_data)):
        preds_for_review = review_predictions.get(i, [])
        if not preds_for_review:
            final = 0
        else:
            final = int(round(sum(preds_for_review) / len(preds_for_review)))
        final_preds.append(final)

        if len(preds_for_review) > 1:
            print(f"Review {i} has chunk predictions: {preds_for_review}")
            print(f"final prediction for review {i}: {final}")
            print(f"true rating: {score_to_class(test_data[i]['rating'])}")
            print("-" * 50)

    true_labels = [score_to_class(r['rating']) for r in test_data]
    final_preds_arr = np.array(final_preds)

    print("First 10 predictions with reviews"+ ":")
    for i in range(min(10, len(test_data))):
        review = test_data[i]
        print(f"Review: {review['review']}")
        print(f"True Class: {true_labels[i]}, Predicted Class: {final_preds[i]}")
        print("-" * 50)

    print("Pred counts: ")
    for i in range(5):
        print(f"  Class {i}: {int((final_preds_arr == i).sum())}")
    print(classification_report(true_labels, final_preds, zero_division=0))

if __name__ == "__main__":
    predict()
