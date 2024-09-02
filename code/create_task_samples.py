import os
import json

from sklearn.model_selection import train_test_split
from datasets import Dataset
from thefuzz import fuzz

from utils.args import create_task_samples_args
from utils import common as c
from utils.ts_creation import deduplicate
from utils.ts_creation import (
    MetaInstructions,
    FormatExtractor,
    generate_few_shots,
    check_prompt_length,
)


parser = create_task_samples_args()
args = parser.parse_args()

if args.task in ["bioqa", "medqa"]:
    prompt_instruction = MetaInstructions.QA_MC_INSTRUCTION
    extract_fn = FormatExtractor.qa_mc
elif args.task == "csqa":
    prompt_instruction = [
        MetaInstructions.QA_YN_INSTRUCTION_Q,
        MetaInstructions.QA_YN_INSTRUCTION_S,
    ]
    extract_fn = FormatExtractor.qa_yn
elif args.task == "recipegen":
    prompt_instruction = MetaInstructions.RECIPEGEN_INSTRUCTION
    extract_fn = FormatExtractor.recipe
elif args.task == "summarization":
    prompt_instruction = MetaInstructions.SUMMARIZATION_INSTRUCTION
    extract_fn = FormatExtractor.summarization
else:
    raise ValueError("Unknown task or no instruction prompt found.")


configs = c.get_configs(args, sampling=True)
model = c.load_vllm_model(args)

few_shots = [fs for fs in c.jsonl_generator(args.few_shot_path, return_string=False)]
corpus_samples = [
    ex for ex in c.jsonl_generator(args.corpus_samples_path, return_string=False)
]

# prepare all few_shot + corpus combinations
prompts = [
    generate_few_shots(
        prompt_instruction=prompt_instruction,
        corpus_example=sample,
        few_shots=few_shots,
        task=args.task,
        num_shots=args.num_shots,
    )
    for sample in corpus_samples
]
prompts = check_prompt_length(args, prompts, max_length=args.max_tokenization_length)
print(f"Number of valid prompts to generate task samples from: {len(prompts)}")

generated = c.vllm_generate(prompts, model, configs["sampling_config"])
task_samples = [{"task_sample": task_sample} for task_sample in generated]

with open(args.output_path_raw, "w") as f:
    for sample in task_samples:
        f.write(json.dumps(sample) + "\n")
print(f"Finished saving {len(task_samples)} raw, unfiltered task samples.")

print("Starting filtering and cleaning of task samples...")
valid_task_samples = []
format_errors = ["index,exception\n"]
for i, sample in enumerate(task_samples):
    try:
        valid_task_samples.append(extract_fn(sample))
    except Exception as e:
        format_errors.append(f"{i},{e}\n")
        continue
print(
    f"Removed {len(task_samples) - len(valid_task_samples)} samples due to formatting errors."
)

with open(args.output_path_error_msgs, "w") as csvfile:
    csvfile.writelines(format_errors)
print("Saved extraction format error messages as a CSV file.")


# two-step fuzzy deduplication
# step 1: filter out task samples that are too similar to human few-shots
few_shot_strings = [extract_fn(s, is_few_shot=True) for s in few_shots]
filtered_1 = [
    s
    for s in valid_task_samples
    if max(fuzz.token_set_ratio(s, fss) for fss in few_shot_strings)
    < args.deduplication_ratio
]
len_filtered_1 = len(filtered_1)
print(
    f"Removed {len(valid_task_samples) - len_filtered_1} samples due to similarity with few-shots."
)

# step 2: deduplicate task samples among themselves
os.environ["TOKENIZERS_PARALLELISM"] = "0"
filtered_2 = deduplicate(filtered_1, ratio=args.deduplication_ratio)
print(
    f"\nRemoved {len_filtered_1 - len(filtered_2)} samples due to similarity among themselves."
)

with open(args.output_path_clean, "w") as f:
    for sample in filtered_2:
        f.write(json.dumps(sample) + "\n")
print(f"Finished saving {len(filtered_2)} clean and filtered task samples.")

num_final = args.num_final_task_samples - len(few_shots)
filtered_3, _ = train_test_split(filtered_2, train_size=num_final)
final_task_samples = [
    {**extract_fn(s, return_dict=True), "is_few_shot": 0} for s in filtered_3  # type: ignore
]
final_task_samples += [
    {**extract_fn(fs, return_dict=True), "is_few_shot": 1} for fs in few_shot_strings  # type: ignore
]

ds = Dataset.from_list(final_task_samples)
ds.save_to_disk(args.output_path_final)
print(f"Finished saving {len(final_task_samples)} final task samples.")
