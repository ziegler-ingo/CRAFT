import os
import sys
import json
from datasets import load_dataset, concatenate_datasets


# https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl
DATASET_NAME = (
    "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
)
CACHE_DIR = sys.argv[1]
CLEAN_STACKEXCHANGE_DIR = CACHE_DIR + "cleaned/"
if not os.path.exists(CLEAN_STACKEXCHANGE_DIR):
    os.makedirs(CLEAN_STACKEXCHANGE_DIR)
    print(f"Directory '{CLEAN_STACKEXCHANGE_DIR}' created.")
else:
    print(f"Directory '{CLEAN_STACKEXCHANGE_DIR}' already exists.")

SUBSETS = [
    "apple",
    "english",
    "codereview",
    "dba",
    "mathoverflow",
    "electronics",
    "mathematica",
    "drupal",
    "magento",
    "gaming",
    "ell",
    "gamedev",
    "gis",
    "askubuntu",
    "diy",
    "academia",
    "blender",
    "cs",
    "chemistry",
    "judaism",
    "crypto",
    "android",
    "ja",
    "christianity",
    "graphicdesign",
    "aviation",
    "ethereum",
    "biology",
    "datascience",
    "law",
    "dsp",
    "japanese",
    "hermeneutics",
    "bicycles",
    "arduino",
    "history",
    "bitcoin",
    "cooking",
    "hinduism",
    "codegolf",
    "boardgames",
    "emacs",
    "economics",
    "gardening",
    "astronomy",
    "islam",
    "german",
    "fitness",
    "french",
    "anime",
    "craftcms",
    "cstheory",
    "engineering",
    "buddhism",
    "linguistics",
    "ai",
    "expressionengine",
    "cogsci",
    "chinese",
    "chess",
    "civicrm",
    "literature",
    "interpersonal",
    "health",
    "avp",
    "earthscience",
    "joomla",
    "homebrew",
    "expatriates",
    "latin",
    "matheducators",
    "ham",
    "genealogy",
    "3dprinting",
    "elementaryos",
    "bioinformatics",
    "devops",
    "hsm",
    "italian",
    "computergraphics",
    "martialarts",
    "bricks",
    "freelancing",
    "crafts",
    "lifehacks",
    "cseducators",
    "materials",
    "hardwarerecs",
    "iot",
    "eosio",
    "languagelearning",
    "korean",
    "coffee",
    "esperanto",
    "beer",
    "ebooks",
    "iota",
    "cardano",
    "drones",
    "conlang",
    "pt",
    "stats",
    "unix",
    "physics",
    "tex",
    "serverfault",
    "salesforce",
    "wordpress",
    "softwareengineering",
    "scifi",
    "security",
    "ru",
    "superuser",
    "sharepoint",
    "rpg",
    "travel",
    "worldbuilding",
    "meta",
    "workplace",
    "ux",
    "money",
    "webmasters",
    "raspberrypi",
    "photo",
    "music",
    "philosophy",
    "puzzling",
    "movies",
    "quant",
    "politics",
    "space",
    "mechanics",
    "skeptics",
    "rus",
    "writers",
    "webapps",
    "softwarerecs",
    "networkengineering",
    "parenting",
    "scicomp",
    "sqa",
    "sitecore",
    "vi",
    "spanish",
    "pm",
    "pets",
    "sound",
    "reverseengineering",
    "outdoors",
    "tridion",
    "retrocomputing",
    "robotics",
    "quantumcomputing",
    "sports",
    "russian",
    "opensource",
    "woodworking",
    "patents",
    "tor",
    "ukrainian",
    "opendata",
    "monero",
    "sustainability",
    "portuguese",
    "mythology",
    "musicfans",
    "or",
    "poker",
    "windowsphone",
    "moderators",
    "stackapps",
    "stellar",
    "vegetarianism",
    "tezos",
]


def cat_cols(batch):
    title_body, upvoted_answer = batch["title_body"], batch["upvoted_answer"]
    cat_batch = [
        tb.strip() + "\n\n" + ua.strip() for tb, ua in zip(title_body, upvoted_answer)
    ]
    return {"text": cat_batch}


print("Starting to load all subsets of the dataset...")
all_datasets = [
    load_dataset(DATASET_NAME, subset, cache_dir=CACHE_DIR) for subset in SUBSETS
]
stackexchange = concatenate_datasets([ds["train"] for ds in all_datasets])  # type: ignore
print("Finished loading all subsets of the dataset.")

print("Preprocessing the dataset...")
stackexchange = stackexchange.map(cat_cols, batched=True).remove_columns(
    ["title_body", "upvoted_answer"]
)
stackexchange = stackexchange.filter(lambda x: 200 <= len(x["text"]) <= 25_000)
print("Finished preprocessing the dataset...")

print("Saving the preprocessed dataset...")
# save in batches of 350k samples
out_path_template = CLEAN_STACKEXCHANGE_DIR + "stack_{:02d}.jsonl"
FILE_SIZE = 350_000
i, file_idx = 0, 0
out_path = out_path_template.format(file_idx)

while i < len(stackexchange):
    batch = stackexchange[i : i + FILE_SIZE]["text"]

    with open(out_path, "a", encoding="utf-8") as f:
        for sample in batch:
            f.write(json.dumps({"text": sample}) + "\n")
            print(f"Saved file stack_{file_idx:02d}.jsonl", end="\r")

    i += FILE_SIZE
    file_idx += 1
    out_path = out_path_template.format(file_idx)
print("\nFinished saving the preprocessed dataset.")
