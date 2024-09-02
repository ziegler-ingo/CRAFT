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
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

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

if args.task == "bioqa":
    # 800 samples
    val_dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    val_dataset = val_dataset.filter(lambda x: x["topic"] == "biology")

    if args.is_llama_3_chat_official:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_llama_chat_prompt(
                    instruction=FormatExtractor.qa_mc(x, eval_format_official=True),
                    system_prompt=c.SystemPrompts.BIOQA_INSTRUCTION,
                )
            },
            load_from_cache_file=False,
        )
    else:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_mistral_inst(
                    instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
                )
                # + "\n"
            },
            load_from_cache_file=False,
        )
elif args.task == "medqa":
    # 4183 samples
    val_dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    val_dataset = val_dataset.map(
        lambda x: {"choices": [x["opa"], x["opb"], x["opc"], x["opd"]]},
        load_from_cache_file=False,
    )
    val_dataset = val_dataset.rename_column("cop", "answer")

    if args.is_llama_3_chat_official:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_llama_chat_prompt(
                    instruction=FormatExtractor.qa_mc(x, eval_format_official=True),
                    system_prompt=c.SystemPrompts.MEDQA_INSTRUCTION,
                )
            },
            load_from_cache_file=False,
        )
    else:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_mistral_inst(
                    instruction=FormatExtractor.qa_mc(x, eval_format_official=True)
                )
                # + "\n"
            },
            load_from_cache_file=False,
        )
elif args.task == "csqa":
    # 2541 samples
    val_dataset = load_dataset("tasksource/commonsense_qa_2.0", split="validation")

    if args.is_llama_3_chat_official:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_llama_chat_prompt(
                    instruction=FormatExtractor.qa_yn(x, eval_format_official=True),
                    system_prompt=c.SystemPrompts.CSQA_INSTRUCTION,
                )
            },
            load_from_cache_file=False,
        )
    else:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_mistral_inst(
                    instruction=FormatExtractor.qa_yn(x, eval_format_official=True)
                )
                # + "\n"
            },
            load_from_cache_file=False,
        )
    val_dataset = val_dataset.rename_column("answer", "answer_str")
    val_dataset = val_dataset.map(
        lambda x: {"answer": 0 if x["answer_str"] == "yes" else 1},
        load_from_cache_file=False,
    )
    val_dataset = val_dataset.add_column("choices", [["yes", "no"]] * len(val_dataset))
elif args.task == "recipegen":
    # 1000 samples
    if args.is_llama_3_chat_official:
        val_dataset = [
            c.create_llama_chat_prompt(
                instruction=FormatExtractor.recipe(x, eval_format_official=True),
                system_prompt=c.SystemPrompts.RECIPEGEN_INSTRUCTION,
            )
            for x in c.jsonl_generator("datasets/recipenlg/test.jsonl")
        ]
    else:
        val_dataset = [
            c.create_mistral_inst(
                instruction=FormatExtractor.recipe(x, eval_format_official=True)
            )
            # + "\n"
            for x in c.jsonl_generator("datasets/recipenlg/test.jsonl")
        ]
    response = [
        f"Ingredients:\n{x['ingredients']}\nSteps:\n{x['steps']}"
        for x in c.jsonl_generator("datasets/recipenlg/test.jsonl")
    ]
    val_dataset = Dataset.from_dict({"text": val_dataset, "response": response})
elif args.task == "summarization":
    # 1000 random samples from available 11,490 samples
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    _, val_idxs = train_test_split(
        range(len(dataset)), test_size=1000, random_state=1234
    )
    val_dataset = dataset.select(val_idxs)

    if args.is_llama_3_chat_official:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_llama_chat_prompt(
                    instruction=FormatExtractor.summarization(
                        x, eval_format_official=True
                    ),
                    system_prompt=c.SystemPrompts.SUMMARIZATION_INSTRUCTION,
                )
            },
            load_from_cache_file=False,
        )
    else:
        val_dataset = val_dataset.map(
            lambda x: {
                "text": c.create_mistral_inst(
                    instruction=FormatExtractor.summarization(
                        x, eval_format_official=True
                    )
                )
                # + "\n"
            },
            load_from_cache_file=False,
        )
    val_dataset = val_dataset.rename_column("highlights", "response")
else:
    raise ValueError("Unknown task.")
assert "text" in val_dataset.column_names
prompts = val_dataset["text"]
print(f"Number of valid evaluation prompts: {len(prompts)}")


#### *** Evaluate classification outputs *** ####
if args.task in ["bioqa", "medqa", "csqa"]:
    outputs = c.vllm_generate(prompts, model, configs["sampling_config"], raw_out=True)

    predictions = c.get_classifications(
        outputs,
        val_dataset["choices"],
        model.get_tokenizer().get_vocab(),
        task=args.task,
    )

    accuracy = c.get_accuracy(predictions, val_dataset["answer"])
    result = {"accuracy": accuracy}

    with open(args.result_path, "w") as f:
        json.dump(result, f)
    print("Finished saving results.")

    max_choices = ["A", "B", "C", "D", "E", "N/A"]
    val_dataset = val_dataset.map(lambda x: {"letter_answer": max_choices[x["answer"]]})
    letter_predictions = [max_choices[pred] for pred in predictions]
    with open(args.output_path, "w") as jsonl_file:
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
    print("Finished saving outputs.")
else:
    #### *** Save generation outputs *** ####
    # Alpaca eval with Llama 3 70B as annotator model in run_eval_winrate.py
    predictions = c.vllm_generate(prompts, model, configs["sampling_config"])

    with open(args.output_path, "w") as jsonl_file:
        for instruction, prediction, response in zip(
            prompts, predictions, val_dataset["response"]
        ):
            sample = {
                "instruction": instruction,
                "prediction": prediction,
                "reference": response,
            }
            json_obj = json.dumps(sample)
            jsonl_file.write(json_obj + "\n")
    print("Finished saving outputs.")

script_end = time()
print(f"Total script time: {script_end - script_start}")
