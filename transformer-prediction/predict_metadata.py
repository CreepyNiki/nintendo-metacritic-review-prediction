import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Pfade werden definiert.
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model_with_metadata")

# Modell, dass gefinetuned wurde.
MODEL_BASE = "roberta-base"

# mit GPU trainieren lassen wenn möglich.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Funktion, welche aufgrund der gegebenen Daten die majority baseline berechnet.
def majority_baseline(y_true):
    # Bestimmt die am häufigsten vorkommende Klasse.
    most_common = Counter(y_true).most_common(1)[0][0]
    y_pred_majority = [most_common] * len(y_true)
    # Berechnung auf Basis, dass alle Instanzen einer Klasse zugeordnet werden.
    print("majority baseline:")
    print(classification_report(y_true, y_pred_majority, zero_division=0))

# Funktion, welche die Confusion Matrix plottet.
def matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Als Heatmap darstellen.
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Funktion, mit welcher die Ergebnisse pro Spiel dargestellt werden können.
def results_per_game(y_true, y_pred):
    # Daten werden geladen.
    test_data = load_json()
    # Hole Spielnamen aus den Daten.
    games = [r.get("game") for r in test_data]

    # Eigene Liste pro Spiel wird erstellt.
    per_game = defaultdict(list)
    # Iterieren über die Spiele, die true Labels und die Predictions. -> Code von Copilot
    for g, t, p in zip(games, y_true, y_pred):
        per_game[g].append((t, p))
        # Loggen der true Labels und Predictions pro Spiel.
    for g, pairs in per_game.items():
        true_labels = [t for t, _ in pairs]
        preds = [p for _, p in pairs]
        print(f"{g}:")
        print(classification_report(true_labels, preds, zero_division=0))

# Kurze Hilfsfunktion, um JSON-Files einzulesen
def load_json():
    path = os.path.join(DATA_DIR, "test/test_with_metadata.json")
    with open(path, encoding='utf-8') as f:
        return json.load(f)

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


def predict():

    # Tokenizer wird geladen
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    model_path = MODEL_DIR
    # Modell wird geladen und GPU wird genutzt, wenn möglich.
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Testdaten werden geladen.
    test_data = load_json()
    print("Test size:", len(test_data))

    chunk_texts = []
    chunk_to_review = []

    # max_length ist hier eine eigene Variable. -> mehrfach benötigt
    max_len = 512

    # spezielle Tokens, die das Modell benötigt, um den Anfang und das Ende eines Textes zu markieren, werden berücksichtigt.
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

    for i, rev in enumerate(test_data):
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

        # Prefix und Review seperat tokenisieren
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        review_ids = tokenizer(review_text, add_special_tokens=False)['input_ids']

        # Berechnung der Anzahl der Tokens, die für den Review-Text in jedem Chunk übrig bleiben, nachdem der Präfix und die speziellen Tokens berücksichtigt wurden.
        chunk_body_size = max_len - len(prefix_ids) - special_tokens

        # Aufteilung des Review-Texts in Chunks, die in das Modell passen. -> Code von ChatGPT
        for start in range(0, len(review_ids), chunk_body_size):
            body_slice = review_ids[start:start + chunk_body_size]
            body_text = tokenizer.decode(body_slice, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # Hinzufügen des Präfixes zu jedem Chunk.
            chunk_text = prefix + " " + body_text
            chunk_texts.append(chunk_text)
            chunk_to_review.append(i)

    # Die Texte werden in Tokens umgewandelt.
    # Diesmal ohne return_overflowing_tokens, da die Chunks bereits in der richtigen Länge zuvor manuell erstellt wurden.
    encodings = tokenizer(chunk_texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")

    # Die encodings werden auf die GPU verschoben, wenn möglich.
    encodings = {k: v.to(device) for k, v in encodings.items()}

    # Gradient wird nicht berechnet. -> schneller
    with torch.no_grad():
        # Vorhersagen für die Chunks berechnen.
        outputs = model(**encodings)
        logits = outputs.logits
        # Predictions in numpy Array umwandeln.
        chunk_preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Dictionary, um die Vorhersagen der Chunks den Reviews zuzuordnen.
    # Key: Index der Review, Value: Liste der Vorhersagen der Chunks, die zu dieser Review gehören.
    # Ided und Grundgerüst der Umsetzung von Claude
    review_predictions = defaultdict(list)
    # Iteration über die Liste.
    for chunk_idx, review_idx in enumerate(chunk_to_review):
        # Klassenvorhersage für einen Chunk.
        pred = int(chunk_preds[chunk_idx])
        # Angehängt an die Liste der Vorhersagen für die entsprechende Review.
        review_predictions[review_idx].append(pred)

    final_preds = []
    # Iterieren über die Reviews.
    for i in range(len(test_data)):
        # Vorhersagen der Chunks für die aktuelle Review.
        preds_for_review = review_predictions.get(i, [])
        # Für jedes Review wird die finale Vorhersage berechnet, indem der Durchschnitt der Vorhersagen der Chunks gebildet und auf die nächste ganze Zahl gerundet wird.
        final = int(round(sum(preds_for_review) / len(preds_for_review)))
        final_preds.append(final)

        # # Loggen der verschiedenen Chunks und dessen Predictions und der finalen Prediction und vergleich zur eigentlichen Prediction.
        # if len(preds_for_review) > 1:
        #     print(f"Review {i} has chunk predictions: {preds_for_review}")
        #     print(f"final prediction for review {i}: {final}")
        #     print(f"true rating: {score_to_class(test_data[i]['rating'])}")
        #     print("-" * 50)

    # True_Labels werden extrahiert und in richtige Format mit 5 Kategorien gebracht.
    true_labels = [score_to_class(r['rating']) for r in test_data]
    # final_preds in numpy Array umwandeln.
    final_preds_arr = np.array(final_preds)

    # # Erste 10 Vorhersagen mit den Reviews loggen.
    # print("First 10 predictions with reviews:")
    # for i in range(min(10, len(test_data))):
    #     review = test_data[i]
    #     print(f"Review: {review['review']}")
    #     print(f"True Class: {true_labels[i]}, Predicted Class: {final_preds[i]}")
    #     print("-" * 50)

    # Übersicht, wie oft welche Klasse predictet wurde.
    print("Pred counts:")
    for i in range(5):
        print(f"  Class {i}: {int((final_preds_arr == i).sum())}")

    # Classification Report
    print("Total:")
    print(classification_report(true_labels, final_preds, zero_division=0))
    results_per_game(true_labels, final_preds)
    # Hilfsfunktionen zur Darstellung der Ergebnisse
    majority_baseline(true_labels)
    matrix(true_labels, final_preds)
    print("MAE: " + mean_absolute_error(true_labels, final_preds))


if __name__ == "__main__":
    predict()
