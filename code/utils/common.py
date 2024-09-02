import os
import json
from time import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft.tuners.lora import LoraConfig

from vllm import LLM, SamplingParams


COMPUTE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def create_llama_chat_prompt(
    instruction,
    system_prompt,
    start_header="<|start_header_id|>",
    end_header="<|end_header_id|>\n\n",
    eot_id="<|eot_id|>",
):
    """
    Create the instruction and system prompt format for Llama 3 models as described here:
    https://github.com/meta-llama/llama-recipes
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3
    https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L222
    """
    return f"""\
{start_header}system{end_header}\
{system_prompt}{eot_id}{start_header}user{end_header}\
{instruction}{eot_id}{start_header}assistant{end_header}\
"""


def create_mistral_inst(
    instruction,
    b_inst="[INST]",
    e_inst="[/INST]",
):
    return f"{b_inst} {instruction.strip()} {e_inst}"


class SystemPrompts:
    BIOQA_INSTRUCTION = """\
You are an expert biologist. \
Select the correct answer from the questions below. \
Answer only with the corresponding letter label.\
"""

    MEDQA_INSTRUCTION = """\
You are a medical expert. \
Select the correct answer from the questions below. \
Answer only with the corresponding letter label.\
"""

    CSQA_INSTRUCTION = """\
You are an expert in common-sense reasoning. \
Select the correct answer from the questions below. \
Answer only with the corresponding letter label.\
"""

    RECIPEGEN_INSTRUCTION = """\
You are an expert chef. \
Directly answer what is asked without additional opening and closing remarks.\
"""

    SUMMARIZATION_INSTRUCTION = """\
You are an expert summarizer. \
Directly answer what is asked without additional opening and closing remarks.\
"""


def get_classifications(outputs, choices, vocab, task):
    if task == "csqa":
        max_choices = ["A", "B"]
    elif task == "medqa":
        max_choices = ["A", "B", "C", "D"]
    elif task == "bioqa":
        max_choices = ["A", "B", "C", "D", "E"]
    else:
        raise ValueError("Unknown task.")

    logprob_idxs = [vocab[letter] for letter in max_choices]
    choices_idxs = [logprob_idxs[: len(choice)] for choice in choices]

    clf = []
    for output, idxs in zip(outputs, choices_idxs):
        if output.outputs[0].text == "":
            print("EMPTY STRING!!!")
            print("*" * 100)
            clf.append(-1)
            continue
        logprobs_idx = [output.outputs[0].logprobs[0].get(idx, None) for idx in idxs]
        logprobs = [x.logprob if x is not None else -float("inf") for x in logprobs_idx]
        if all(x == -float("inf") for x in logprobs):
            clf.append(-1)
        else:
            clf.append(logprobs.index(max(logprobs)))

    return clf


def get_accuracy(preds, labels):
    correct = sum(pred == lbl for pred, lbl in zip(preds, labels))
    total = len(labels)

    return correct / total


def vllm_generate(prompts, model, sampling_config, raw_out=False):
    outputs = model.generate(prompts, sampling_config, use_tqdm=True)
    outputs_text = [sample.outputs[0].text.strip() for sample in outputs]

    return outputs if raw_out else outputs_text


def check_hf_token(path):
    try:
        with open(path, "r") as f:
            hf_token = f.read().strip()
            print("Loaded Hugging Face token.")
    except (FileNotFoundError, AttributeError):
        hf_token = None
        print("Hugging Face token not loaded.")

    return hf_token


def jsonl_generator(file_path, return_string=False):
    with open(file_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            if not return_string:
                yield sample
            else:
                out = "\n\n".join(f"{k}: {v}" for k, v in sample.items())
                yield out


def get_directory_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def load_model_and_tokenizer(args, add_eos_token=None):
    hf_token = (
        check_hf_token(args.hf_token_path) if hasattr(args, "hf_token_path") else None
    )

    start = time()
    print("Starting loading of model via transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=COMPUTE_DTYPES["bfloat16"],
        device_map="auto" if not hasattr(args, "merge_adapters") else "cpu",
        trust_remote_code=False,
        cache_dir=args.cache_dir,
        return_dict=hasattr(args, "merge_adapters"),
        token=hf_token,
    )
    load_time = time() - start
    print(f"Finished loading of model via transformers in {load_time:.2f} seconds.")
    print(f"Loaded model in data type: {next(iter(model.parameters())).dtype}.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
        add_eos_token=add_eos_token,
        token=hf_token,
    )

    return model, tokenizer


def load_vllm_model(args, temp_dir=None):
    if temp_dir is not None:
        model_name_or_path = temp_dir
    else:
        model_name_or_path = args.model_name_or_path

    max_model_len = getattr(args, "max_model_len", None)
    if max_model_len is not None:
        max_model_len = min(max_model_len, 8192)
    else:
        max_model_len = 8192 if "llama" in model_name_or_path else 16384

    start = time()
    print("Starting loading of model via vllm...")
    model = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        trust_remote_code=False,
        tensor_parallel_size=torch.cuda.device_count(),
        swap_space=0,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_logprobs=getattr(args, "logprobs", None),
    )
    load_time = time() - start
    print(f"Finished loading of model via vllm in {load_time:.2f} seconds.")

    return model


def get_configs(
    args,
    generation=False,
    sampling=False,
    lora=False,
):
    configs = {}

    if generation:
        generation_config = GenerationConfig(
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        configs["generation_config"] = generation_config
    else:
        configs["generation_config"] = None

    if sampling:
        sampling_config = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_new_tokens,
            logprobs=getattr(args, "logprobs", None),
        )
        configs["sampling_config"] = sampling_config
    else:
        configs["sampling_config"] = None

    if lora:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        configs["lora_config"] = lora_config
    else:
        configs["lora_config"] = None

    return configs
