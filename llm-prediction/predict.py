import http.client
import json
from dotenv import load_dotenv
import os
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import random
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Aufrufen .env File.
load_dotenv()

# Pfade werden definiert.
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')

# Seed für zufällige Auswahl der Testdaten.
seed = 42

# Funktion, welche aufgrund der gegebenen Daten die majority baseline berechnet.
def majority_baseline(y_true):
    # Bestimmt die am häufigsten vorkommende Klasse.
    most_common = Counter(y_true).most_common(1)[0][0]
    y_pred_majority = [most_common] * len(y_true)
    # Berechnung auf Basis, dass alle Instanzen einer Klasse zugeordnet werden.
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
def results_per_game(y_true, y_pred, metadata, reviews):

    # Hole Spielnamen aus den Daten.
    games = [r.get("game") for r in reviews]

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

# Hilfsfunktion, um JSON-Files einzulesen
def load_json(metadata):
    if metadata:
        json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
    else:
        json_path = os.path.join(DATA_DIR, 'all_without_metadata.json')
    with open(json_path, encoding='utf-8') as f:
        return json.load(f)

# Hilfsfunktion, die die Reviews aus den JSON-Dateien extrahiert. Die Reviews werden in einer Liste zurückgegeben.
def prepare(items):
    return [
        review
        for reviews in items.values()
        if isinstance(reviews, list)
        for review in reviews
    ]

# Funktion, die die Review-Text und das Datum oder die Metadaten in einem String zusammenfasst. Das Datum und die Metadaten werden hierbei vor den Reviewtext geschrieben.
def prepareData(review, metadata):
    # Wenn Metadaten vorhanden sind, werden diese mit in den Prompt aufgenommen.
    if metadata:
        m = review["metadata"]
        return f"""
        Date: {review['date']}
        AverageUserScore: {m['averageUserScore']}
        GamesReviewed: {m['scoreCounts']['positive']}
        PositiveReviews: {m['scoreCounts']['positive']}
        NeutralReviews: {m['scoreCounts']['neutral']}
        NegativeReviews: {m['scoreCounts']['negative']}
        Review: {review['review']}
        """
    else:
        # Wenn keine Metadaten vorhanden sind, wird nur das Datum und der Reviewtext in den Prompt aufgenommen.
        return f"""
        Date: {review['date']}
        Review: {review['review']}
        """

# Funktion, die den numerischen Score in eine von 5 Klassen umwandelt. Dabei werden die Scores in folgende Klassen eingeteilt:
# 0: Score 0-1 -> sehr schlechte Bewertung
# 1: Score 2-3 -> schlechte Bewertung
# 2: Score 4-6 -> durchschnittliche Bewertung
# 3: Score 7-8 -> gute Bewertung
# 4: Score 9-10 -> sehr gute Bewertung
def score_to_class(score):
    try:
        s = int(score)
    except Exception:
        return 0
    if s <= 1:
        return 0
    elif s <= 3:
        return 1
    elif s <= 6:
        return 2
    elif s <= 8:
        return 3
    else:
        return 4

# Hauptfunktion, welche mit dem Modell kommuniziert und dessen Antworten sammelt.
def useModel(metadata, size, few_shot):
    # Korrektes JSON-File wird ausgewählt.
    # Daten werden geladen und vorbereitet
    items = load_json(metadata)
    reviews = prepare(items)
    total_reviews = len(reviews)

    # Die Anzahl der Reviews, die für die Vorhersage genutzt werden, wird bestimmt. Es werden entweder alle Reviews oder eine zufällige Auswahl von Reviews genutzt, abhängig von der Size, die als Parameter übergeben wird.
    k = min(size, total_reviews)
    # Zufällige Auswahl von k Reviews aus den verfügbaren Reviews, basierend auf dem Seed werden ausgewählt.
    # Die Anzahl basiert auf dem Size Parameter.
    rnd = random.Random(seed)
    selected_indices = rnd.sample(range(total_reviews), k)

    # Ausgewählte Reviews werden in neuer Liste gespeichert.
    reviews = [reviews[i] for i in selected_indices]

    # Loggen von Anzahl an Reviews.
    print(f"Loaded {len(reviews)} reviews (from {total_reviews} available)")

    # Daten werden in die richtige Form gebracht.
    texts = [prepareData(r, metadata) for r in reviews]

    # True_Labels werden extrahiert und in richtige Format mit 5 Kategorien gebracht.
    true_labels = [score_to_class(r.get('rating', 0)) for r in reviews]

    # API-Key wird aus den Umgebungsvariablen geladen und Verbindung zum API-Endpunkt wird hergestellt.
    api_key = os.getenv("API_KEY")
    conn = http.client.HTTPSConnection("chat.kiconnect.nrw")

    # Few-Shot-Prompting
    if few_shot:
        system_prompt = (
            "You are an expert to predict the score of a game review. "
            "You will be given a review and you have to predict the score of the review on a scale from 0 to 10. Return the score as an int number. "
            "You should only output the score and nothing else."
            "Here are some examples:"
            "Review: 'Worst game I ever played. Save your money and don't buy it. Story is dull, combat is clunky as hell."
            "True Class: 0"
            "Review: 'Well this not the best Pokemon ever made, but it has clearly lots of potential for fun, and is at least interesting, not gonna talk about the dlc tho....'"
            "True Class: 2"
            "Review: 'What an amazing game. If this released on S2 hardware the reviews would be very different. One of the best Pokémon experiences I have had.'"
            "True Class: 4"
        )
    # Standard-Prompt
    else:
        system_prompt = (
            "You are an expert to predict the score of a game review. "
            "You will be given a review and you have to predict the score of the review on a scale from 0 to 10. Return the score as an int number. "
            "You should only output the score and nothing else."
        )

    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer " + api_key
    }

    preds = []
    # Hinzufügen der Texte zur Payload.
    # Auswahl des Modells.
    for idx, text in enumerate(texts):
        payload = {
            "model": "Openai GPT OSS 120B",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0

        }
        # # Den Prompt loggen, um zu sehen, wie er aussieht.
        # print(text)

        # Payload wird in JSON-Format umgewandelt und in UTF-8 kodiert.
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        # Request an das Modell wird gesendet.
        conn.request("POST", "/api/v1/chat/completions", body, headers)
        # Antwort wird gelesen und dekodiert.
        res = conn.getresponse()
        data = res.read().decode('utf-8')

        # Antwort wird in JSON-Format umgewandelt.
        j = json.loads(data)
        # Die Antwort des Modells wird extrahiert.

        # Diese Zahl wird dann in die entsprechende Klasse umgewandelt und in der Liste preds gespeichert.
        c0 = j['choices'][0]
        content = c0['message'].get('content')
        # Regex wird verwendet, um die Zahl aus der Antwort zu extrahieren.
        match = re.search(r"\d+", content)
        # Zahl wird in richtige Format mit 5 Kategorien gebracht.
        match = score_to_class(match.group())
        # Vorhersage wird in Liste gespeichert.
        preds.append(match)

        # Alle 10 Reviews wird geloggt, wie viele Reviews bereits verarbeitet wurden.
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(texts)} reviews")

    # # Erste 10 Vorhersagen mit den Reviews loggen.
    # print("First 10 reviews with predictions und true labels:")
    # for i in range(min(10, len(reviews))):
    #     review = reviews[i]
    #     print(f"Review: {review['review']}")
    #     print(f"True Class: {true_labels[i]}, Predicted Class: {preds[i]}")
    #     print("-" * 50)

    # Übersicht, wie oft welche Klasse predictet wurde.
    print("Pred counts:")
    for i in range(5):
        print(f"  Class {i}: {preds.count(i)}")

    # Classification Report
    print(classification_report(true_labels, preds, zero_division=0))
    results_per_game(true_labels, preds, metadata, reviews)
    # Hilfsfunktionen zur Darstellung der Ergebnisse
    majority_baseline(true_labels)
    matrix(true_labels, preds)
    print(mean_absolute_error(true_labels, preds))

if __name__ == "__main__":
    useModel(False, 100, False)