import os
import sys
import json

from tqdm import tqdm


WIKIPEDIA_PATH = sys.argv[1]  # path where all the subfiles are
CLEAN_WIKIPEDIA_PATH = WIKIPEDIA_PATH + "cleaned/"

if not os.path.exists(CLEAN_WIKIPEDIA_PATH):
    os.makedirs(CLEAN_WIKIPEDIA_PATH)
    print(f"Directory '{CLEAN_WIKIPEDIA_PATH}' created.")
else:
    print(f"Directory '{CLEAN_WIKIPEDIA_PATH}' already exists.")


# clean the samples, cut at 30 newline-delimited paragraphs
# cutting off at paragraphs is fine here as information in these documents isn't sequential
cleaned_texts = []
uid = 0
for wiki_file in tqdm(os.listdir(WIKIPEDIA_PATH), desc="files", file=sys.stdout):
    if not (wiki_file.startswith("wiki_") and wiki_file.endswith(".jsonl")):
        continue
    else:
        with open(WIKIPEDIA_PATH + wiki_file, "r") as f:
            for line in f:
                json_article = json.loads(line)
                paragraphs = json_article["text"].strip().split("\n")[:30]
                text = " ".join(paragraphs)
                length = len(text)

                if 200 <= length <= 25_000:
                    cleaned_texts.append(text)
    # store only files with at least 350k samples to minimize IO
    if len(cleaned_texts) >= 350_000:
        with open(CLEAN_WIKIPEDIA_PATH + f"wiki_{uid:02d}.jsonl", "w") as f:
            for text in cleaned_texts:
                f.write(json.dumps({"text": text}) + "\n")
        cleaned_texts.clear()
        uid += 1
