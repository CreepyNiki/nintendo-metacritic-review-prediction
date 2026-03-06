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
            full_path = os.path.join(root, file)
            with open(full_path, encoding='utf-8') as f:
                j = json.load(f)

            if file.endswith('_with_metadata.json') or file.endswith('-with_metadata.json'):
                    merged_with.update(j)
            else:
                    merged_without.update(j)

    out_with = os.path.join(DATA_DIR, 'all_with_metadata.json')
    out_without = os.path.join(DATA_DIR, 'all_without_metadata.json')

    with open(out_with, 'w', encoding='utf-8') as f:
        json.dump(merged_with, f, ensure_ascii=False, indent=2)

    with open(out_without, 'w', encoding='utf-8') as f:
        json.dump(merged_without, f, ensure_ascii=False, indent=2)

    # print(f"Wrote {len(merged_with)} entries to {out_with}")
    # print(f"Wrote {len(merged_without)} entries to {out_without}")
