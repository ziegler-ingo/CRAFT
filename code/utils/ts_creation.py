import os
import json
import random
from concurrent.futures import ProcessPoolExecutor

from thefuzz import fuzz

from utils import common as c


def generate_few_shots(
    prompt_instruction,
    corpus_example,
    few_shots,
    task,
    num_shots=3,
    b_inst="[INST]",
    e_inst="[/INST]",
    eos="</s>",
):
    """
    Create a few-shot prompt in the format described at:
    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
    """
    out = ""  # tokenizer adds bos token at the start

    # CSQA: we sample either question or statement instruction
    if type(prompt_instruction) is list:
        prompt_instruction = random.choice(prompt_instruction)
    indices = random.sample(range(len(few_shots)), num_shots)
    for idx in indices:
        shot = few_shots[idx]
        text = shot["Text"].strip()
        if task in ["bioqa", "medqa"]:
            json_dict = FormatExtractor.qa_mc(shot, is_few_shot=True)
        elif task == "csqa":
            json_dict = FormatExtractor.qa_yn(shot, is_few_shot=True)
        elif task == "recipegen":
            json_dict = FormatExtractor.recipe(shot, is_few_shot=True)
        elif task == "summarization":
            json_dict = FormatExtractor.summarization(shot, is_few_shot=True)
        else:
            raise ValueError("Unknown task.")

        formatted = (
            f"{b_inst} {prompt_instruction} \n\n___________\nText: {text} {e_inst} "
        )
        formatted += f"{json_dict}{eos} "
        out += formatted

    out += f"{b_inst} {prompt_instruction} \n\n___________\nText: {corpus_example['text'].strip()} {e_inst} "

    return out


def check_prompt_length(args, prompts, max_length=4096):
    """
    Filter out combined few-shot and corpus example prompt above a set tokenized length.
    """
    hf_token = (
        c.check_hf_token(args.hf_llama_2_token_path)
        if hasattr(args, "hf_llama_2_token_path")
        else None
    )

    tokenizer = c.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if hasattr(args, "cache_dir") else None,
        use_fast=True,
        token=hf_token,
    )

    tokenized = tokenizer(prompts, return_length=True, return_attention_mask=False)
    lengths = tokenized["length"]

    valid_prompts = [p for p, l in zip(prompts, lengths) if l <= max_length]

    return valid_prompts


class MetaInstructions:
    # used for BioQA, MedQA
    QA_MC_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one question along with four answer choices designated as A, B, C, and D based on the provided text. \
Then, respond to the question with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"question": "<question here>", "options": ["A. <option A here>", "B. <option B here>", "C. <option C here>", "D. <option D here>"], "answer": "<letter label of correct answer here>"}\
"""

    # used for common-sense QA
    QA_YN_INSTRUCTION_Q = """\
Please carefully read the text below. \
Then, generate exactly one question that is answerable with yes or no based on the provided text. \
Then, respond to the question with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"question": "<question here>", "options": ["A. Yes", "B. No"], "answer": "<letter label of correct answer here>"}\
"""

    # used for common-sense QA
    QA_YN_INSTRUCTION_S = """\
Please carefully read the text below. \
Then, generate exactly one statement that is answerable with yes or no based on the provided text. \
Then, respond to the statement with the correct answer using only the corresponding letter label. \
Return the output only as a JSON structure in this format: \
{"statement": "<statement here>", "options": ["A. Yes", "B. No"], "answer": "<letter label of correct answer here>"}\
"""

    RECIPEGEN_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one short one-sentence instruction to prepare the dish named in the text. \
Then, generate a detailed recipe for the dish by listing the required ingredients and steps. \
Return the output only as a JSON structure in this format: \
{"instruction": "<short one-sentence cooking instruction here>", "ingredients": ["<ingredient one here>", "<ingredient two here>", "< ... >", "<continue until end>"], "steps": ["<step one here>", "<step two here>", "< ... >", "<continue until end>"]}\
"""

    SUMMARIZATION_INSTRUCTION = """\
Please carefully read the text below. \
Then, generate exactly one instruction for summarizing the text below. \
Then, generate the summary. \
Afterwards, extract a long but clean version of the initial text that encompasses the summary, but keeps almost all details from the initial text. \
Return the output only as a JSON structure in this format: \
{"instruction": "<summary instruction here>", "summary": "<summary here>", "long_but_clean_text": "<long but clean text that encompasses the summary but keeps almost all details from the initial text here>"} \
"""


class FormatExtractor:
    @staticmethod
    def qa_mc(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean BioQA and MedQA multiple choice task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = sample["question"] + "\n" + "\n".join(sample["options"])
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["answer"]
            out = inst + sample["answer"]
            return out
        elif finetune_format_official:
            question = sample["question"]
            max_choices = ["A", "B", "C", "D", "E"]
            choices = sample["choices"]
            n_choices = len(choices)
            answers = "\n".join(
                [
                    f"{choice}. {text}"
                    for choice, text in zip(max_choices[:n_choices], choices)
                ]
            )
            inst = question + "\n" + answers
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + max_choices[sample["answer"]]
            out = inst + max_choices[sample["answer"]]
            return out
        elif eval_format_official:
            question = sample["question"]
            max_choices = ["A", "B", "C", "D", "E"]
            choices = sample["choices"]
            n_choices = len(choices)
            answers = "\n".join(
                [
                    f"{choice}. {text}"
                    for choice, text in zip(max_choices[:n_choices], choices)
                ]
            )
            out = question + "\n" + answers
            return out
        else:
            # if any valid JSON is in output string, get it
            # if multiple valid JSONs exist, the first one is returned
            question, options, answer = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    qa_dict = json.loads(json_str)
                    question = qa_dict["question"].strip()
                    options = [opt.strip() for opt in qa_dict["options"]]
                    answer = qa_dict["answer"][0].strip()

                    if len(question) <= 15:
                        question = ""
                        raise QuestionLengthError(
                            "No question  found / below 15 characters."
                        )
                    elif len(options) != 4:
                        options = ""
                        raise OptionsNumberError(
                            "Number of answer options other than 4."
                        )
                    elif answer not in "ABCD":
                        answer = ""
                        raise AnswerFormatError(
                            "Answer does not include A, B, C, or D."
                        )
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([question, options, answer]), f"{error_msgs[-1]}"

            out_dict = {"question": question, "options": options, "answer": answer}
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def qa_yn(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean common sense QA yes-no task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = sample["Instruction"].split("\n")
            options = [option.strip() for option in instruction_parts[1:] if option]
            output = sample["Output"]
            out_dict = {
                "question": instruction_parts[0].strip(),
                "options": options,
                "answer": output[0].strip(),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = sample["question"] + "\n" + "\n".join(sample["options"])
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["answer"]
            out = inst + sample["answer"]
            return out
        elif finetune_format_official:
            letter_to_idx = {"yes": 0, "no": 1}
            question = sample["question"]
            max_choices = ["A", "B"]
            choices = ["yes", "no"]
            answers = "\n".join(
                [f"{choice}. {text}" for choice, text in zip(max_choices, choices)]
            )
            inst = question + "\n" + answers
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + max_choices[letter_to_idx[sample["answer"]]]
            out = inst + max_choices[letter_to_idx[sample["answer"]]]
            return out
        elif eval_format_official:
            letter_to_idx = {"yes": 0, "no": 1}
            question = sample["question"]
            max_choices = ["A", "B"]
            choices = ["yes", "no"]
            answers = "\n".join(
                [f"{choice}. {text}" for choice, text in zip(max_choices, choices)]
            )
            out = question + "\n" + answers
            return out
        else:
            # if any valid JSON is in output string, get it
            # if multiple valid JSONs exist, the first one is returned
            question, options, answer = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    qa_dict = json.loads(json_str)
                    if "question" in qa_dict:
                        question = qa_dict["question"].strip()
                    elif "statement" in qa_dict:
                        question = qa_dict["statement"].strip()
                    options = [opt.strip() for opt in qa_dict["options"]]
                    answer = qa_dict["answer"][0].strip()

                    if len(question) <= 15:
                        question = ""
                        raise QuestionLengthError(
                            "No question or statement found / shorter than 15 characters."
                        )
                    elif len(options) != 2:
                        options = ""
                        raise OptionsNumberError(
                            "Number of answer options other than 2."
                        )
                    elif answer not in "AB":
                        answer = ""
                        raise AnswerFormatError("Answer does not include A or B.")
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([question, options, answer]), f"{error_msgs[-1]}"

            out_dict = {"question": question, "options": options, "answer": answer}
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def recipe(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean recipe task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction = sample["Instruction"].strip()
            output_parts = sample["Output"].split("Steps:")
            out_dict = {
                "instruction": instruction,
                "ingredients": [
                    ing.strip() for ing in output_parts[0].split("\n")[1:] if ing
                ],
                "steps": [step.strip() for step in output_parts[1].split("\n") if step],
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            # format final samples loaded from HF dataset object
            inst = c.create_mistral_inst(sample["instruction"])
            ingredients = "\n".join(sample["ingredients"])
            steps = "\n".join(sample["steps"])
            # out = f"{inst}\nIngredients:\n{ingredients}\nSteps:\n{steps}"
            out = f"{inst}Ingredients:\n{ingredients}\nSteps:\n{steps}"
            return out
        elif finetune_format_official:
            inst = c.create_mistral_inst(sample["instruction"])
            ingredients = sample["ingredients"]
            steps = sample["steps"]
            # out = f"{inst}\nIngredients:\n{ingredients}\nSteps:\n{steps}"
            out = f"{inst}Ingredients:\n{ingredients}\nSteps:\n{steps}"
            return out
        elif eval_format_official:
            out = sample["instruction"]
            return out
        else:
            instruction, ingredients, steps = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    recipe_dict = json.loads(json_str)
                    instruction = recipe_dict["instruction"].strip()
                    ingredients = [
                        ing.strip() for ing in recipe_dict["ingredients"] if ing
                    ]
                    steps = [step.strip() for step in recipe_dict["steps"] if step]

                    if len(instruction) <= 15:
                        instruction = ""
                        raise QuestionLengthError(
                            "No instruction found / below 15 characters."
                        )
                    elif not type(ingredients) is list:
                        ingredients = ""
                        raise AnswerFormatError("Ingredients not in list format.")
                    elif len(ingredients) < 1:
                        ingredients = ""
                        raise AnswerFormatError("One or no ingredient found.")
                    elif not type(steps) is list:
                        steps = ""
                        raise AnswerFormatError("Steps not in list format.")
                    elif len(steps) < 1:
                        steps = ""
                        raise AnswerFormatError("One or no cooking step found.")
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([instruction, ingredients, steps]), f"{error_msgs[-1]}"

            out_dict = {
                "instruction": instruction,
                "ingredients": ingredients,
                "steps": steps,
            }
        out_json = json.dumps(out_dict)

        return out_json

    @staticmethod
    def summarization(
        sample,
        is_few_shot=False,
        return_dict=False,
        finetune_format=False,
        finetune_format_official=False,
        eval_format_official=False,
    ):
        """
        Filter and clean summarization task samples.
        """
        if is_few_shot:
            # no checks needed because few-shots have correct format by design
            instruction_parts = [
                part.strip() for part in sample["Instruction"].split("\n") if part
            ]
            out_dict = {
                "instruction": instruction_parts[0],
                "summary": sample["Output"].strip(),
                "long_but_clean_text": "\n".join(instruction_parts[1:]),
            }
        elif return_dict:
            # already cleaned task samples
            return json.loads(sample)
        elif finetune_format:
            inst = f"{sample['instruction']}\n{sample['long_but_clean_text']}"
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["summary"]
            out = inst + sample["summary"]
            return out
        elif finetune_format_official:
            inst = f"Please summarize the text below:\n{sample['article']}"
            inst = c.create_mistral_inst(inst)
            # out = inst + "\n" + sample["highlights"]
            out = inst + sample["highlights"]
            return out
        elif eval_format_official:
            out = f"Please summarize the text below:\n{sample['article']}"
            return out
        else:
            instruction, summary, long_but_clean_text = "", "", ""
            sample_str = sample["task_sample"]
            assert "}" in sample_str, "No JSON found."

            error_msgs = []
            potential_jsons = sample_str.split("}")
            for potential_json in potential_jsons:
                if not potential_json:
                    continue
                json_str = potential_json + "}"

                try:
                    summary_dict = json.loads(json_str)
                    instruction = summary_dict["instruction"].strip()
                    summary = summary_dict["summary"].strip()
                    long_but_clean_text = summary_dict["long_but_clean_text"].strip()

                    if len(instruction) <= 15:
                        instruction = ""
                        raise QuestionLengthError(
                            "Instruction shorter than 15 characters."
                        )
                    elif len(summary) <= 100:
                        summary = ""
                        raise AnswerFormatError("Summary shorter than 100 characters.")
                    elif len(long_but_clean_text) <= 500:
                        long_but_clean_text = ""
                        raise AnswerFormatError(
                            "Long but clean text shorter than 500 characters."
                        )
                    break
                except Exception as e:
                    error_msgs.append(e)
                    continue

            assert all([instruction, summary, long_but_clean_text]), f"{error_msgs[-1]}"

            out_dict = {
                "instruction": instruction,
                "summary": summary,
                "long_but_clean_text": long_but_clean_text,
            }
        out_json = json.dumps(out_dict)

        return out_json


class QuestionLengthError(Exception):
    pass


class OptionsNumberError(Exception):
    pass


class AnswerFormatError(Exception):
    pass


def partition(seq, num_partitions):
    if num_partitions >= len(seq):
        return [seq]
    else:
        avg = len(seq) // num_partitions
        remainder = len(seq) % num_partitions
        partitions = [
            seq[i * avg + min(i, remainder) : (i + 1) * avg + min(i + 1, remainder)]
            for i in range(num_partitions)
        ]
        return partitions


def compare_samples(s1, seen_partition, ratio):
    for s2 in seen_partition:
        if s1 == s2:
            continue
        sim_ratio = fuzz.token_set_ratio(s1, s2)
        if sim_ratio >= ratio:
            return True
    return False


def deduplicate(seqs, ratio):
    avail_cpus = os.cpu_count()
    assert avail_cpus is not None
    num_partitions = avail_cpus - 1
    len_seqs = len(seqs)
    duplicate_idxs = set()
    seen_samples = set()

    with ProcessPoolExecutor() as executor:
        for i, s1 in enumerate(seqs, start=1):
            print(f"Comparing file {i:,} of {len_seqs:,}", end="\r")
            if s1 in seen_samples:
                continue
            seen_samples.add(s1)
            partitions = partition(list(seen_samples), num_partitions)
            s1_duplicated = [s1] * len(partitions)
            ratio_duplicated = [ratio] * len(partitions)

            if any(
                executor.map(
                    compare_samples, s1_duplicated, partitions, ratio_duplicated
                )
            ):
                seen_samples.remove(s1)
                duplicate_idxs.add(i - 1)

    deduplicated = [seqs[i] for i in range(len_seqs) if i not in duplicate_idxs]
    return deduplicated
