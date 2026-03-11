import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "prediction_transformer/models/model_without_metadata_5class")
MODEL_DIR_WITH_METADATA = os.path.join(ROOT, "prediction_transformer/models/model_with_metadata_5class")

MODEL_BASE = "roberta-base"

def load_json(metadata):
    if metadata:
        path = os.path.join(DATA_DIR, "test/test_with_metadata.json")
    else:
        path = os.path.join(DATA_DIR, "test/test_without_metadata.json")
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def prepareData(review, metadata):
    if metadata:
        m = review["metadata"]
        return f"""
        Review: {review['review']}
        Date: {review['date']}
        AverageUserScore: {m['averageUserScore']}
        GamesReviewed: {m['games']}
        PositiveReviews: {m['scoreCounts']['positive']}
        NeutralReviews: {m['scoreCounts']['neutral']}
        NegativeReviews: {m['scoreCounts']['negative']}
        """
    else:
        return f"""
        Review: {review['review']}
        Date: {review['date']}
        """

# Score → 5-Klassen Mapping
def score_to_class(score):
    score = int(score)
    if score <= 1:
        return 0  # sehr schlecht
    elif score <= 3:
        return 1  # schlecht
    elif score <= 6:
        return 2  # mittel
    elif score <= 8:
        return 3  # gut
    else:  # 9-10
        return 4  # sehr gut

def predict(metadata=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    model_path = MODEL_DIR_WITH_METADATA if metadata else MODEL_DIR
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_data = load_json(metadata)
    print("Test size:", len(test_data))

    texts = [prepareData(r, metadata) for r in test_data]
    encodings = tokenizer(
        texts, truncation=True, max_length=512, padding=True, return_tensors="pt"
    )

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # True Labels in 5 Klassen
    true_labels = [score_to_class(r['rating']) for r in test_data]

    print("First 10 predictions with reviews" + (" and metadata" if metadata else "") + ":")
    for i in range(min(10, len(test_data))):
        review = test_data[i]
        print(f"Review: {review['review']}")
        print(f"True Class: {true_labels[i]}, Predicted Class: {preds[i]}")
        print("-" * 50)


    print("Pred counts" + (" with metadata:" if metadata else ":"))
    for i in range(5):
        print(f"  Class {i}: {sum(preds == i)}")
    print("Classification Report (5 Klassen: 0=sehr schlecht ... 4=sehr gut):")
    print(classification_report(true_labels, preds, zero_division=0))

if __name__ == "__main__":
    predict(False)
    # predict(True)