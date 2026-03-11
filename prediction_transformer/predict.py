import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, mean_absolute_error

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

GOLDSTANDARD_PATH = os.path.join(ROOT, "model_with_metadata")
MODEL_DIR = os.path.join(ROOT, "prediction_transformer/models/model_without_metadata")
MODEL_DIR_WITH_METADATA = os.path.join(ROOT, "prediction_transformer/models/model_with_metadata")
DATA_DIR = os.path.join(ROOT, "data")

MODEL_BASE = "xlm-roberta-base"

def load_json(metadata):
    if(metadata):
        path = os.path.join(DATA_DIR, "test/test_with_metadata.json")
    else:
        path = os.path.join(DATA_DIR, "test/test_without_metadata.json")
    with open(path, encoding='utf-8') as f:
        j = json.load(f)
        return j

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

def predict(metadata=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    if(metadata):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_WITH_METADATA)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.to(device)
    model.eval()

    test_data = load_json(metadata)

    print("test size: " + str(len(test_data)))

    texts = [prepareData(review, metadata) for review in test_data]

    encodings = tokenizer(
        texts, truncation=True, max_length=512, padding=True, return_tensors="pt")

    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    predictions = predictions.astype(int)

    print("First 10 predictions with reviews" + (" and metadata" if metadata else "") + ":")
    for i in range(min(10, len(test_data))):
        review = test_data[i]
        print(f"Review: {review['review']}")
        print(f"True Rating: {review['rating']}, Predicted Rating: {predictions[i]}")
        print("-" * 50)

    true_labels = [int(review['rating']) for review in test_data]

    print("Classification Report:")
    print(classification_report(true_labels, predictions, zero_division=0))

if __name__ == "__main__":
    predict(True)
    predict(False)
