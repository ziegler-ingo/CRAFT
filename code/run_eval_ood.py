"""
Note:
We trained both Mistral and Llama models to have Mistral style INST patterns.
Therefore, only when we evaluate the official Llama 3 models, we need the
Llama 3 prompt style.
"""

import json
from time import time
import tempfile

import torch
from datasets import load_dataset

import safetensors.torch
from peft.peft_model import PeftModel

from utils import common as c
from utils.args import evaluate_model_args
from utils.ts_creation import FormatExtractor


parser = evaluate_model_args()
args = parser.parse_args()
print(args)

script_start = time()

if args.merge_adapters:
    #### *** Start of adapter loading and merging *** ####
    temp_dir = tempfile.TemporaryDirectory(dir="models/")
    temp_dir_path = temp_dir.name
    modified = False

    start = time()
    model, tokenizer = c.load_model_and_tokenizer(args)
    print(
        f"Loaded model for merging: {args.model_name_or_path}, initial model dtype: {model.dtype}."
    )
    time_taken = time() - start
    print(f"Time taken for model loading: {time_taken:.2f}")

    print(f"Starting to load adapters: {args.adapters_path}.")
    adapter_tensors = safetensors.torch.load_file(
        args.adapters_path + "adapter_model.safetensors",
        device="cpu",
    )
    non_lora_keys = [k for k in adapter_tensors.keys() if "lora" not in k]

    if non_lora_keys:
        for nlk in non_lora_keys:
            del adapter_tensors[nlk]
            print(f"Removed {nlk} from {args.adapters_path}.")
        modified = True
    else:
        print("No non-lora-keys found in adapters.")

    # check data type of adapters
    dtype = next(iter(adapter_tensors.values())).dtype
    print(f"Adapters are in dtype {dtype}.")
    if dtype is not torch.bfloat16:
        adapter_tensors = {
            k: v.to(c.COMPUTE_DTYPES["bfloat16"]) for k, v in adapter_tensors.items()
        }
        print("Casted adapters to bfloat16.")
        modified = True

    if modified:
        safetensors.torch.save_file(
            adapter_tensors, args.adapters_path + "adapter_model.safetensors"
        )
        print(f"Saved modified adapters at {args.adapters_path}.")
    del adapter_tensors

    print("Combining and merging model and adapters...")
    model = PeftModel.from_pretrained(
        model=model,
        model_id=args.adapters_path,
        torch_dtype=c.COMPUTE_DTYPES["bfloat16"],
    )
    model = model.merge_and_unload()
    print("Finished combining and merging adapters.")

    print("Saving merged model and tokenizer...")
    model.save_pretrained(temp_dir_path)
    tokenizer.save_pretrained(temp_dir_path)
    print(f"Finished saving merged model and tokenizer at temporary output directory.")
    del model, tokenizer
    torch.cuda.empty_cache()
else:
    temp_dir_path = None


#### *** Start of output generation *** ####
configs = c.get_configs(args, sampling=True)
print("Start loading merged model for inference.")
model = c.load_vllm_model(args, temp_dir=temp_dir_path)
print("Loaded merged model for inference.")


### medical genetics
medical_genetics = load_dataset("tasksource/mmlu", "medical_genetics", split="test")
val_dataset = medical_genetics.map(
    lambda x: {
        "text": c.create_mistral_inst(
            instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
        )
    },
    load_from_cache_file=False,
)
assert "text" in val_dataset.column_names
prompts = val_dataset["text"]
print(f"Number of valid evaluation prompts, medical_genetics: {len(prompts)}")

outputs = c.vllm_generate(prompts, model, configs["sampling_config"], raw_out=True)

predictions = c.get_classifications(
    outputs,
    val_dataset["choices"],
    model.get_tokenizer().get_vocab(),
    task=args.task,
)

accuracy = c.get_accuracy(predictions, val_dataset["answer"])
result = {"accuracy-medical_genetics": accuracy}

max_choices = ["A", "B", "C", "D", "E", "N/A"]
val_dataset = val_dataset.map(lambda x: {"letter_answer": max_choices[x["answer"]]})
letter_predictions = [max_choices[pred] for pred in predictions]
with open(args.output_path + "-medical_genetics", "w") as jsonl_file:
    for instruction, prediction, reference in zip(
        prompts, letter_predictions, val_dataset["letter_answer"]
    ):
        sample = {
            "instruction": instruction,
            "prediction": prediction,
            "reference": reference,
        }
        json_obj = json.dumps(sample)
        jsonl_file.write(json_obj + "\n")
print("Finished saving medical_genetics outputs.")


### anatomy
anatomy = load_dataset("tasksource/mmlu", "anatomy", split="test")
val_dataset = anatomy.map(
    lambda x: {
        "text": c.create_mistral_inst(
            instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
        )
    },
    load_from_cache_file=False,
)
assert "text" in val_dataset.column_names
prompts = val_dataset["text"]
print(f"Number of valid evaluation prompts, anatomy: {len(prompts)}")

outputs = c.vllm_generate(prompts, model, configs["sampling_config"], raw_out=True)

predictions = c.get_classifications(
    outputs,
    val_dataset["choices"],
    model.get_tokenizer().get_vocab(),
    task=args.task,
)

accuracy = c.get_accuracy(predictions, val_dataset["answer"])
result["accuracy-anatomy"] = accuracy

val_dataset = val_dataset.map(lambda x: {"letter_answer": max_choices[x["answer"]]})
letter_predictions = [max_choices[pred] for pred in predictions]
with open(args.output_path + "-anatomy", "w") as jsonl_file:
    for instruction, prediction, reference in zip(
        prompts, letter_predictions, val_dataset["letter_answer"]
    ):
        sample = {
            "instruction": instruction,
            "prediction": prediction,
            "reference": reference,
        }
        json_obj = json.dumps(sample)
        jsonl_file.write(json_obj + "\n")
print("Finished saving anatomy outputs.")


### high_school_biology
high_school_biology = load_dataset(
    "tasksource/mmlu", "high_school_biology", split="test"
)
val_dataset = high_school_biology.map(
    lambda x: {
        "text": c.create_mistral_inst(
            instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
        )
    },
    load_from_cache_file=False,
)
assert "text" in val_dataset.column_names
prompts = val_dataset["text"]
print(f"Number of valid evaluation prompts, high_school_biology: {len(prompts)}")

outputs = c.vllm_generate(prompts, model, configs["sampling_config"], raw_out=True)

predictions = c.get_classifications(
    outputs,
    val_dataset["choices"],
    model.get_tokenizer().get_vocab(),
    task=args.task,
)

accuracy = c.get_accuracy(predictions, val_dataset["answer"])
result["accuracy-high_school_biology"] = accuracy

val_dataset = val_dataset.map(lambda x: {"letter_answer": max_choices[x["answer"]]})
letter_predictions = [max_choices[pred] for pred in predictions]
with open(args.output_path + "-high_school_biology", "w") as jsonl_file:
    for instruction, prediction, reference in zip(
        prompts, letter_predictions, val_dataset["letter_answer"]
    ):
        sample = {
            "instruction": instruction,
            "prediction": prediction,
            "reference": reference,
        }
        json_obj = json.dumps(sample)
        jsonl_file.write(json_obj + "\n")
print("Finished saving high_school_biology outputs.")


### college_biology
college_biology = load_dataset("tasksource/mmlu", "college_biology", split="test")
val_dataset = college_biology.map(
    lambda x: {
        "text": c.create_mistral_inst(
            instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
        )
    },
    load_from_cache_file=False,
)
assert "text" in val_dataset.column_names
prompts = val_dataset["text"]
print(f"Number of valid evaluation prompts, college_biology: {len(prompts)}")

outputs = c.vllm_generate(prompts, model, configs["sampling_config"], raw_out=True)

predictions = c.get_classifications(
    outputs,
    val_dataset["choices"],
    model.get_tokenizer().get_vocab(),
    task=args.task,
)


accuracy = c.get_accuracy(predictions, val_dataset["answer"])
result["accuracy-college_biology"] = accuracy

val_dataset = val_dataset.map(lambda x: {"letter_answer": max_choices[x["answer"]]})
letter_predictions = [max_choices[pred] for pred in predictions]
with open(args.output_path + "-college_biology", "w") as jsonl_file:
    for instruction, prediction, reference in zip(
        prompts, letter_predictions, val_dataset["letter_answer"]
    ):
        sample = {
            "instruction": instruction,
            "prediction": prediction,
            "reference": reference,
        }
        json_obj = json.dumps(sample)
        jsonl_file.write(json_obj + "\n")
print("Finished saving college_biology outputs.")

# only save results once at the end after all tasks
with open(args.result_path, "w") as f:
    json.dump(result, f)
print("Finished saving results.")


script_end = time()
print(f"Total script time: {script_end - script_start}")
