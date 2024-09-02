import json
from functools import partial

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from vllm import SamplingParams

from utils import common as c
from utils.args import evaluate_model_args
from utils.alpaca import basic_prompt, basic_parser


parser = evaluate_model_args()
args = parser.parse_args()
prompt_template = partial(basic_prompt, prompt_kind="llama")


if args.task == "recipegen":
    instructions, responses = [], []
    for sample in c.jsonl_generator("datasets/recipenlg/test.jsonl"):
        instructions.append(sample["instruction"])
        responses.append(
            f"Ingredients:\n{sample['ingredients']}\nSteps:\n{sample['steps']}"
        )

    df = pd.DataFrame(
        {
            "instruction": instructions,
            "output_1": responses,
            "output_2": [
                out["prediction"] for out in c.jsonl_generator(args.output_path)
            ],
        }
    )
    np.random.seed(1234)
    mask = np.random.choice([True, False], size=len(df))
    df.loc[mask, ["output_1", "output_2"]] = df.loc[
        mask, ["output_2", "output_1"]
    ].values
    df["swapped"] = mask
    df["prompt"] = df.apply(prompt_template, axis=1)
    prompts = df["prompt"].tolist()
elif args.task == "summarization":
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    _, val_idxs = train_test_split(
        range(len(dataset)), test_size=1000, random_state=1234
    )
    df = dataset.select(val_idxs).to_pandas()
    df["instruction"] = df.apply(
        lambda x: f"Please summarize the text below:\n{x['article']}", axis=1
    )
    synth_outputs = [out["prediction"] for out in c.jsonl_generator(args.output_path)]
    df["output_2"] = synth_outputs
    df = df.rename(columns={"highlights": "output_1"})
    np.random.seed(1234)
    mask = np.random.choice([True, False], size=len(df))
    df.loc[mask, ["output_1", "output_2"]] = df.loc[
        mask, ["output_2", "output_1"]
    ].values
    df["swapped"] = mask
    df["prompt"] = df.apply(prompt_template, axis=1)
    prompts = df["prompt"].tolist()
else:
    raise ValueError("Unknown task.")
assert prompts, "Prompt list is empty or does not exist."
print("Number of evaluation prompts: ", len(prompts))

llama_3_70b_path = "models/hf_models/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/e8cf5276ae3e97cfde8a058e64a636f2cde47820"
model = c.load_vllm_model(args, temp_dir=llama_3_70b_path)
sampling_config = SamplingParams(temperature=0.0, top_p=1, top_k=-1, max_tokens=50)
outputs = c.vllm_generate(prompts, model, sampling_config=sampling_config)
df["model_preference"] = [basic_parser(o) for o in outputs]
df.to_json(f"{args.output_path_annotator}.json")

win_rate = (
    (df["swapped"] & (df["model_preference"] == 1))
    | (~df["swapped"] & (df["model_preference"] == 2))
).mean()
result = {"win_rate": win_rate}

with open(args.result_path, "w") as f:
    json.dump(result, f)
print("Finished saving results.")
