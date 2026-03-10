import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, mean_absolute_error

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_DIR = os.path.join(ROOT, "prediction_transformer/models/model_without_metadata")
DATA_DIR = os.path.join(ROOT, "data")

MODEL_BASE = "xlm-roberta-base"

def load_json(metadata=False):
    if metadata:
        path = os.path.join(DATA_DIR, 'test_with_metadata.json')
    else:
        path = os.path.join(DATA_DIR, 'test_without_metadata.json')
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

def predict(metadata=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
    ).to(device)

    model.eval()

    test_data = load_json(metadata)

    print("Preparing texts...")
    texts = [prepareData(review, metadata) for review in test_data]

    print("Tokenizing...")
    encodings = tokenizer(
        texts, truncation=True, max_length=512, padding=True, return_tensors="pt")

    encodings = {k: v.to(device) for k, v in encodings.items()}

    print("Predicting...")
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    predictions = predictions * 10

    print("Sample predictions:", predictions[:10])

    true_labels = [int(review['rating']) for review in test_data]

    print("Classification Report:")
    print(classification_report(true_labels, predictions, digits=4))

if __name__ == "__main__":
    predict(False)
