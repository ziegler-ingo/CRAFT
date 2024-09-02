import os
import sys
import csv
import json


WIKIHOW_PATH = sys.argv[1]
WIKIHOW_FILE = WIKIHOW_PATH + "wikihowAll.csv"
WIKIHOW_TEST_TITLES = sys.argv[2]
CLEAN_WIKIHOW_PATH = WIKIHOW_PATH + "cleaned/"

if not os.path.exists(CLEAN_WIKIHOW_PATH):
    os.makedirs(CLEAN_WIKIHOW_PATH)
    print(f"Directory '{CLEAN_WIKIHOW_PATH}' created.")
else:
    print(f"Directory '{CLEAN_WIKIHOW_PATH}' already exists.")


with open(WIKIHOW_TEST_TITLES, "r", encoding="utf-8") as f:
    test_set_titles = {line.strip() for line in f}

train_set, test_set = [], []
i = 1
with open(WIKIHOW_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    _header = next(reader)

    for row in reader:
        try:
            headline, title, text = row
        except ValueError:
            print(f"Skipping row {i} due to wrong format.")
            continue

        print(f"Processing line {i}", end="\r")
        title = title.strip()
        _title = "".join(l for l in title if l.isalnum())

        if _title in test_set_titles:
            test_set.append({"instruction": title, "reference": headline.strip()})
        # do not filter on paragraphs first since we only want full documents due to sequential nature
        elif 200 <= len(text.strip()) <= 25_000:
            clean_text = title + "\n"
            clean_text += text.strip()
            train_set.append({"text": clean_text})
        i += 1

print("\nSaving train files...")
with open(CLEAN_WIKIHOW_PATH + "train.jsonl", "w", encoding="utf-8") as f:
    for text_dict in train_set:
        f.write(json.dumps(text_dict) + "\n")
print("Saved train files.")

print("Saving test files...")
with open(CLEAN_WIKIHOW_PATH + "test.jsonl", "w", encoding="utf-8") as f:
    for text_dict in test_set:
        f.write(json.dumps(text_dict) + "\n")
print("Saved test files.")
