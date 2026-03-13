import http.client
import json
from dotenv import load_dotenv
import os
from sklearn.metrics import classification_report
import random
import re

load_dotenv()

seed = 42

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def prepare(items):
    return [
        review
        for reviews in items.values()
        if isinstance(reviews, list)
        for review in reviews
    ]

def prepareData(review, metadata):
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
        return f"""
        Date: {review['date']}
        Review: {review['review']}
        """

def score_to_class(score):
    """Map raw numeric score (likely 0-10) to classes 0..4."""
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

def useModel(metadata, size, few_shot):
    if metadata:
        json_path = os.path.join(DATA_DIR, 'all_with_metadata.json')
    else:
        json_path = os.path.join(DATA_DIR, 'all_without_metadata.json')

    items = load_json(json_path)
    reviews = prepare(items)
    total_reviews = len(reviews)

    k = min(size, total_reviews)
    rnd = random.Random(seed)
    selected_indices = rnd.sample(range(total_reviews), k)

    reviews = [reviews[i] for i in selected_indices]

    print(f"Loaded {len(reviews)} reviews (from {total_reviews} available)")

    texts = [prepareData(r, metadata) for r in reviews]

    true_labels = [score_to_class(r.get('rating', 0)) for r in reviews]

    api_key = os.getenv("API_KEY")
    conn = http.client.HTTPSConnection("chat.kiconnect.nrw")

    if few_shot:
        system_prompt = (
            "You are an expert to predict the score of a game review. Return the score as an integer."
            "Predict the score from 0 to 4. (0 = very bad, 1 = bad, 2 = neutral, 3 = good, 4 = very good)"
            "You should only output the score and nothing else."
            "Here are some examples:"
            "Review: 'Worst game I ever played. Save your money and don't buy it. Story is dull, combat is clunky as hell.: 0"
            "Review: 'Well this not the best Pokemon ever made, but it has clearly lots of potential for fun, and is at least interesting, not gonna talk about the dlc tho....', True Class: 2"
            "Review: 'What an amazing game. If this released on S2 hardware the reviews would be very different. One of the best Pokémon experiences I have had.', True Class: 4"
        )
        # system_prompt = (
        #     "You are an expert to predict the score of a game review. "
        #     "You will be given a review and you have to predict the score of the review on a scale from 0 to 10. Return the score as an int number. "
        #     "You should only output the score and nothing else."
        #     "Here are some examples:"
        #     "Review: 'Worst game I ever played. Save your money and don't buy it. Story is dull, combat is clunky as hell.: 0"
        #     "Review: 'Well this not the best Pokemon ever made, but it has clearly lots of potential for fun, and is at least interesting, not gonna talk about the dlc tho....', True Class: 2"
        #     "Review: 'What an amazing game. If this released on S2 hardware the reviews would be very different. One of the best Pokémon experiences I have had.', True Class: 4"
        # )
    else:
        system_prompt = (
            "You are an expert to predict the score of a game review. Return the score as an integer."
            "Predict the score from 0 to 4. (0 = very bad, 1 = bad, 2 = neutral, 3 = good, 4 = very good)"
            "You should only output the score and nothing else."
        )
        # system_prompt = (
        #     "You are an expert to predict the score of a game review. "
        #     "You will be given a review and you have to predict the score of the review on a scale from 0 to 10. Return the score as an int number. "
        #     "You should only output the score and nothing else."
        # )

    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer " + api_key
    }

    preds = []
    for idx, text in enumerate(texts):
        payload = {
            "model": "Openai GPT OSS 120B",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0
        }
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')

        conn.request("POST", "/api/v1/chat/completions", body, headers)
        res = conn.getresponse()
        data = res.read().decode('utf-8')


        j = json.loads(data)
        c0 = j['choices'][0]
        content = c0['message'].get('content')
        match = re.search(r"\d+", content)
        preds.append(int(match[0]))

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(texts)} reviews")

    print("First 20 reviews with predictions und true labels:" + (" and metadata" if metadata else "") + ":")
    for i in range(min(20, len(reviews))):
        review = reviews[i]
        print(f"Review: {review['review']}")
        print(f"True Class: {true_labels[i]}, Predicted Class: {preds[i]}")
        print("-" * 50)

    print("Pred counts:")
    for i in range(5):
        print(f"  Class {i}: {preds.count(i)}")

    print(classification_report(true_labels, preds, zero_division=0))


if __name__ == "__main__":
    useModel(True, 100, False)