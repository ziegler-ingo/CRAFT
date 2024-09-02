import os

from sklearn.model_selection import train_test_split

from datasets import load_dataset, load_from_disk, Dataset
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer
from peft.mapping import get_peft_model

from utils.args import finetune_model_args
from utils.finetune import LoraSavingCallback
from utils.ts_creation import FormatExtractor
from utils import common as c


parser = finetune_model_args()
args = parser.parse_args()
print(args)
set_seed(args.random_seed)
MAX_TRAIN_SAMPLES = 25_000

os.environ["TOKENIZER_PARALLELISM"] = "false"
configs = c.get_configs(args, lora=True)
model, tokenizer = c.load_model_and_tokenizer(args, add_eos_token=True)

# add pad_token to tokenizer, original model was trained without it
# adjust embedding size and pad token IDs accordingly
tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.padding_side = "right"
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.get_input_embeddings().padding_idx = tokenizer.pad_token_id

model = get_peft_model(model, configs["lora_config"])
# cache is only used for inference
model.config.use_cache = False  # type: ignore


if args.task_samples_path == "official":
    if args.task == "bioqa":
        # 2464 samples
        train_dataset = load_dataset("derek-thomas/ScienceQA", split="train")
        train_dataset = train_dataset.filter(lambda x: x["topic"] == "biology")
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.qa_mc(x, finetune_format_official=True)},
            load_from_cache_file=False,
        )
    elif args.task == "medqa":
        # 182,822 samples, but downsampled to 25k
        train_dataset = load_dataset("openlifescienceai/medmcqa", split="train")
        train_dataset = train_dataset.map(
            lambda x: {"choices": [x["opa"], x["opb"], x["opc"], x["opd"]]},
            load_from_cache_file=False,
        )
        train_dataset = train_dataset.rename_column("cop", "answer")
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.qa_mc(x, finetune_format_official=True)},
            load_from_cache_file=False,
        )
    elif args.task == "csqa":
        # 9264 samples
        train_dataset = load_dataset("tasksource/commonsense_qa_2.0", split="train")
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.qa_yn(x, finetune_format_official=True)},
            load_from_cache_file=False,
        )
    elif args.task == "recipegen":
        # 500,000 samples, but downsampled to 25k
        train_dataset = [
            FormatExtractor.recipe(x, finetune_format_official=True)
            for x in c.jsonl_generator("datasets/recipenlg/train.jsonl")
        ]
        train_dataset = Dataset.from_dict({"text": train_dataset})
    elif args.task == "summarization":
        # 287,113 samples, but downsampled to 25k
        train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
        train_dataset = train_dataset.map(
            lambda x: {
                "text": FormatExtractor.summarization(x, finetune_format_official=True)
            },
            load_from_cache_file=False,
        )
    else:
        raise ValueError("Unknown task or no train dataset found.")
else:
    train_dataset = load_from_disk(args.task_samples_path)
    if args.few_shots_only:
        train_dataset = train_dataset.filter(lambda x: x["is_few_shot"] == 1)
    if args.task in ["bioqa", "medqa"]:
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.qa_mc(x, finetune_format=True)},
            load_from_cache_file=False,
        )
    elif args.task == "csqa":
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.qa_yn(x, finetune_format=True)},
            load_from_cache_file=False,
        )
    elif args.task == "recipegen":
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.recipe(x, finetune_format=True)},
            load_from_cache_file=False,
        )
    elif args.task == "summarization":
        train_dataset = train_dataset.map(
            lambda x: {"text": FormatExtractor.summarization(x, finetune_format=True)},
            load_from_cache_file=False,
        )
    else:
        raise ValueError("Unknown task or no instruction prompt found.")
assert "text" in train_dataset.column_names, "`text` column not found."  # type: ignore
if train_dataset.num_rows > MAX_TRAIN_SAMPLES:
    num_rows_before = train_dataset.num_rows
    train_sample_idx, _ = train_test_split(
        range(num_rows_before), train_size=MAX_TRAIN_SAMPLES, random_state=1234
    )
    train_dataset = train_dataset.select(train_sample_idx)
    print(f"Sampled {MAX_TRAIN_SAMPLES} train samples from {num_rows_before}.")

# llama 3 tokenizer does not add eos tokens
if "Llama" in args.model_name_or_path:
    train_dataset = train_dataset.map(
        lambda x: {"text": f"{x['text']}{tokenizer.eos_token}"},
        load_from_cache_file=False,
    )

os.environ["TOKENIZER_PARALLELISM"] = "true"

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    bf16=args.use_bf16,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    gradient_accumulation_steps=args.grad_acc_steps,
    num_train_epochs=args.num_epochs,
    logging_strategy=args.logging_strategy,
    logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,
    save_only_model=args.save_only_model,
    optim=args.optim,
    dataloader_num_workers=args.num_workers,
    save_total_limit=args.save_total_limit,
)
callbacks = [LoraSavingCallback()]

trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=configs["lora_config"],
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=9999,  # filler, max_len is limited by max generated tokens
    tokenizer=tokenizer,
    callbacks=callbacks,
)

trainer.train()
