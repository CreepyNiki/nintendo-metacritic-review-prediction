# python
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
MODEL_DIR_WITH_METADATA = os.path.join(ROOT, "prediction_transformer/models/model_with_metadata")
GOLDSTANDARD_PATH = os.path.join(DATA_DIR, "../model_without_metadata")

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

    # model_path = GOLDSTANDARD_PATH
    model_path = MODEL_DIR_WITH_METADATA if metadata else MODEL_DIR
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_data = load_json(metadata)
    print("Test size:", len(test_data))

    texts = [prepareData(r, metadata) for r in test_data]

    # Tokenize with overflow so lange Reviews in Chunks gesplittet werden
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
        stride=128,
        return_overflowing_tokens=True
    )

    # Mapping von Chunk-Index -> Original-Review-Index
    mapping = encodings.pop("overflow_to_sample_mapping")
    # Behalte eine CPU-Kopie der input_ids zum Decodieren/debuggen
    input_ids_cpu = encodings["input_ids"].clone().cpu()

    # Move rest to device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        chunk_preds = torch.argmax(logits, dim=1).cpu().numpy()  # Vorhersagen pro Chunk (NumPy Array)

    # mapping kann ein Tensor sein -> Liste
    if not isinstance(mapping, list):
        try:
            mapping = mapping.tolist()
        except:
            mapping = list(mapping)

    # Aggregiere Chunk-Vorhersagen pro Review
    review_predictions = defaultdict(list)
    for chunk_idx, sample_idx in enumerate(mapping):
        pred = int(chunk_preds[chunk_idx])
        review_predictions[sample_idx].append(pred)

    # Erzeuge finale Vorhersagen (mittlerer Wert gerundet) in Testdaten-Reihenfolge
    final_preds = []
    for i in range(len(test_data)):
        preds_for_review = review_predictions.get(i, [])
        if not preds_for_review:
            # Falls kein Chunk (sollte selten sein), Standardklasse 0
            final = 0
        else:
            final = int(round(sum(preds_for_review) / len(preds_for_review)))
        final_preds.append(final)

        # Debug-Ausgabe für Reviews mit mehreren Chunks
        if len(preds_for_review) > 1:
            chunk_texts = [
                tokenizer.decode(input_ids_cpu[k], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for k, sid in enumerate(mapping) if sid == i
            ]
            print(f"Chunks: {len(preds_for_review)} for review {i}")
            print(f"Chunk content: {chunk_texts}")
            print(f"Review {i} has chunk predictions: {preds_for_review}")
            print(f"final prediction for review {i}: {final}")
            print(f"True rating: {test_data[i]['rating']} -> class {score_to_class(test_data[i]['rating'])}")
            print("-" * 50)

    true_labels = [score_to_class(r['rating']) for r in test_data]
    final_preds_arr = np.array(final_preds)

    print("First 10 predictions with reviews" + (" and metadata" if metadata else "") + ":")
    for i in range(min(10, len(test_data))):
        review = test_data[i]
        print(f"Review: {review['review']}")
        print(f"True Class: {true_labels[i]}, Predicted Class: {final_preds[i]}")
        print("-" * 50)

    print("Pred counts" + (" with metadata:" if metadata else ":"))
    for i in range(5):
        print(f"  Class {i}: {int((final_preds_arr == i).sum())}")
    print("Classification Report (5 Klassen: 0=sehr schlecht ... 4=sehr gut):")
    print(classification_report(true_labels, final_preds, zero_division=0))

if __name__ == "__main__":
    predict(False)
    # predict(True)
