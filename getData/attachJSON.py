import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.normpath(DATA_DIR)

def convert_lists_to_objects():
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                wrapped = {"reviews": data}
                with open(full_path, 'w', encoding='utf-8') as f:
                    json.dump(wrapped, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":

    merged_with = {}
    merged_without = {}

    convert_lists_to_objects()

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if not file.endswith('.json'):
                continue
            if file in ('all_with_metadata.json', 'all_without_metadata.json'):
                continue

            if(file == 'test_with_metadata.json' or file == 'test_without_metadata.json'):
                continue

            full_path = os.path.join(root, file)
            with open(full_path, encoding='utf-8') as f:
                j = json.load(f)

            reviews = []
            if isinstance(j, list):
                reviews = j
            elif isinstance(j, dict):
                if 'reviews' in j and isinstance(j['reviews'], list):
                    reviews = j['reviews']
                else:
                    lists = [v for v in j.values() if isinstance(v, list)]
                    if lists:
                        reviews = max(lists, key=len)
                    else:
                        reviews = [j]

            name = os.path.splitext(file)[0]
            for suf in ('_with_metadata', '-with_metadata'):
                if name.endswith(suf):
                    name = name[:-len(suf)]
            if file.endswith('_with_metadata.json') or file.endswith('-with_metadata.json'):
                merged_with[name] = reviews
            else:
                merged_without[name] = reviews

    out_with = os.path.join(DATA_DIR, 'all_with_metadata.json')
    out_without = os.path.join(DATA_DIR, 'all_without_metadata.json')

    with open(out_with, 'w', encoding='utf-8') as f:
        json.dump(merged_with, f, ensure_ascii=False, indent=2)

    with open(out_without, 'w', encoding='utf-8') as f:
        json.dump(merged_without, f, ensure_ascii=False, indent=2)

    total_games_with = len(merged_with)
    total_reviews_with = sum(len(v) for v in merged_with.values())
    total_games_without = len(merged_without)
    total_reviews_without = sum(len(v) for v in merged_without.values())


    print(f"Wrote {total_games_with} games ({total_reviews_with} reviews) to {out_with}")
    print(f"Wrote {total_games_without} games ({total_reviews_without} reviews) to {out_without}")
