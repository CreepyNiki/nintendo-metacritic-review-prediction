import json
import os
from collections import OrderedDict

# Pfad für die Daten.
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# Funktion, um Listen in JSON-Dateien in Objekte zu konvertieren -> automatisch generiert von Copilot.
def convert_lists_to_objects():
    # Laden von Daten
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.json'):
                continue
            if (file == 'test_with_metadata.json' or file == 'test_without_metadata.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Umwandeln Listen in Objekte.
            if isinstance(data, list):
                wrapped = {"reviews": data}
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(wrapped, f, ensure_ascii=False, indent=2)

# Funktion, die Spiel in die JSON-Struktur einfügt.
def writeGameIntoJSON(review, game_name):
    out = OrderedDict()
    for k, v in review.items():
        out[k] = v
        if k == 'date':
            out['game'] = game_name
    return out



if __name__ == "__main__":

    merged_with = {}
    merged_without = {}

    # Umwandeln von Listen in Objekte.
    convert_lists_to_objects()

    # Einlesen von JSON-Dateien.
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.json'):
                continue

            # bisherige zusammengefasste Dateien überspringen.
            if file in ('all_with_metadata.json', 'all_without_metadata.json'):
                continue

            # Testdateien nicht miteinbeziehen.
            if (file == 'test_with_metadata.json' or file == 'test_without_metadata.json'):
                continue

            full_path = os.path.join(root, file)
            with open(full_path, encoding='utf-8') as f:
                j = json.load(f)

            reviews = j['reviews']

            name = os.path.splitext(file)[0]
            # je nach Suffix unterschiedliche Ziel-Dictionaries befüllen.
            if name.endswith('_with_metadata'):
                # Benennen von Spielüberschriften
                name = name[:-len('_with_metadata')]
            elif name.endswith('_without_metadata'):
                name = name[:-len('_without_metadata')]
            if file.endswith('_with_metadata.json'):
                merged_with[name] = reviews
            else:
                merged_without[name] = reviews

            enriched_reviews = [writeGameIntoJSON(r, name) for r in reviews]

            if file.endswith('_with_metadata.json'):
                merged_with[name] = enriched_reviews
            else:
                merged_without[name] = enriched_reviews

    # Outputfiles benannt.
    out_with = os.path.join(DATA_DIR, 'all_with_metadata.json')
    out_without = os.path.join(DATA_DIR, 'all_without_metadata.json')

    # Outputfiles rausgeschrieben.
    with open(out_with, 'w', encoding='utf-8') as f:
        json.dump(merged_with, f, ensure_ascii=False, indent=2)

    with open(out_without, 'w', encoding='utf-8') as f:
        json.dump(merged_without, f, ensure_ascii=False, indent=2)

    # Spiele und Reviews zusammengefasst rausgeloggt. -> für Debugging und Übersicht
    total_games_with = len(merged_with)
    total_reviews_with = sum(len(v) for v in merged_with.values())
    total_games_without = len(merged_without)
    total_reviews_without = sum(len(v) for v in merged_without.values())

    print(f"Wrote {total_games_with} games ({total_reviews_with} reviews) to {out_with}")
    print(f"Wrote {total_games_without} games ({total_reviews_without} reviews) to {out_without}")
