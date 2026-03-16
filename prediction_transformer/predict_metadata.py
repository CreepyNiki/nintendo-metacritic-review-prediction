import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "prediction_transformer/models/model_with_metadata")
GOLDSTANDARD_PATH = os.path.join(DATA_DIR, "../model_with_metadata")

MODEL_BASE = "roberta-base"

def majority_baseline(y_true):
    most_common = Counter(y_true).most_common(1)[0][0]
    y_pred_majority = [most_common] * len(y_true)
    print(classification_report(y_true, y_pred_majority, zero_division=0))

def matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def load_json():
    path = os.path.join(DATA_DIR, "test/test_with_metadata.json")
    with open(path, encoding='utf-8') as f:
        return json.load(f)

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

    # model_path = GOLDSTANDARD_PATH
    model_path = MODEL_DIR
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_data = load_json()
    print("Test size:", len(test_data))

    chunk_texts = []
    chunk_to_review = []

    max_len = 512

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    for i, rev in enumerate(test_data):
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

        # Prefix und Review seperat tokenisieren
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        review_ids = tokenizer(review_text, add_special_tokens=False)['input_ids']

        chunk_body_size = max_len - len(prefix_ids) - special_tokens

        for start in range(0, len(review_ids), chunk_body_size):
            body_slice = review_ids[start:start + chunk_body_size]
            body_text = tokenizer.decode(body_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            chunk_text = prefix + " " + body_text
            chunk_texts.append(chunk_text)
            chunk_to_review.append(i)

    encodings = tokenizer(
        chunk_texts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        chunk_preds = torch.argmax(logits, dim=1).cpu().numpy()

    review_predictions = defaultdict(list)
    for chunk_idx, review_idx in enumerate(chunk_to_review):
        pred = int(chunk_preds[chunk_idx])
        review_predictions[review_idx].append(pred)

    final_preds = []
    for i in range(len(test_data)):
        preds_for_review = review_predictions.get(i, [])
        if not preds_for_review:
            final = 0
        else:
            final = int(round(sum(preds_for_review) / len(preds_for_review)))
        final_preds.append(final)

        # if len(preds_for_review) > 1:
        #     print(f"Review {i} has chunk predictions: {preds_for_review}")
        #     print(f"final prediction for review {i}: {final}")
        #     print(f"true rating: {score_to_class(test_data[i]['rating'])}")
        #     print("-" * 50)

    true_labels = [score_to_class(r['rating']) for r in test_data]
    final_preds_arr = np.array(final_preds)

    # print("First 10 predictions with reviews:")
    # for i in range(min(10, len(test_data))):
    #     review = test_data[i]
    #     print(f"Review: {review['review']}")
    #     print(f"True Class: {true_labels[i]}, Predicted Class: {final_preds[i]}")
    #     print("-" * 50)

    print("Pred counts" + (" with metadata:"))
    for i in range(5):
        print(f"  Class {i}: {int((final_preds_arr == i).sum())}")

    print(classification_report(true_labels, final_preds, zero_division=0))
    majority_baseline(true_labels)
    matrix(true_labels, final_preds)
    print(mean_absolute_error(true_labels, final_preds))


if __name__ == "__main__":
    predict()
