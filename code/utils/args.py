import argparse


def create_embeddings_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="Path to read in the data to be embedded."
    )
    parser.add_argument(
        "--group_name",
        type=str,
        help="Name of group in HDF5 database. Should represent the overall dataset name which will be embedded.",
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        default="./",
        help="Path where the embeddings HDF5 database will be stored.",
    )
    parser.add_argument(
        "--saving_file",
        type=str,
        default="embeddings.h5",
        help="Filename of the embeddings HDF5 database.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="Model to use to embed texts. Must be included in SentenceTransformers package.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        help="Compression type to use for HDF5 database.",
    )
    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        help="GZIP compression level to use for HDF5 database.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Whether to use a GPU or process on CPU.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size to use during sample embedding process.",
    )
    parser.add_argument(
        "--use_amp_fp16",
        default=False,
        action="store_true",
        help="Whether or not to run inference and embedding storing in fp16 or fp32.",
    )

    return parser


def retrieve_docs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database_path",
        type=str,
        help="Path where the embeddings HDF5 database is stored.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        help="Path to file where the few-shot samples are stored.",
    )
    parser.add_argument(
        "--c4_path",
        type=str,
        help="Path where the C4 corpus is stored.",
    )
    parser.add_argument(
        "--stackexchange_path",
        type=str,
        help="Path where the StackExchange corpus is stored.",
    )
    parser.add_argument(
        "--wikipedia_path",
        type=str,
        help="Path where the Wikipedia corpus is stored.",
    )
    parser.add_argument(
        "--wikihow_path",
        type=str,
        help="Path where the WikiHow corpus is stored.",
    )
    parser.add_argument(
        "--saving_path",
        type=str,
        default="./",
        help="Path where the retrieved documents will be stored.",
    )
    parser.add_argument(
        "--top_p_percentile",
        type=float,
        default=0.05,
        help="Top-p percentile of documents to include for the global similarity comparison to few-shot examples.",
    )
    parser.add_argument(
        "--num_samples_to_retrieve",
        type=int,
        help=(
            "Number of similar documents to retrieve from the embedding database. "
            "Should ideally be divisible by number of available few-shot examples."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="multi-qa-MiniLM-L6-cos-v1",
        help="Model to use to embed texts. Must be included in SentenceTransformers package.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Whether to use a GPU or process on CPU.",
    )
    parser.add_argument(
        "--use_amp_fp16",
        default=False,
        action="store_true",
        help="Whether or not to run inference and embedding storing in fp16 or fp32.",
    )

    return parser


def evaluate_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Hugging Face model name or path to evaluate.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path where downloaded model is stored.",
    )
    parser.add_argument(
        "--adapters_path",
        type=str,
        help="Path to checkpoint containing the adapters to load.",
    )
    parser.add_argument(
        "--merge_adapters",
        default=False,
        action="store_true",
        help="Flag to activate merging. Should always be included when running the merge_adapters.py script.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "bioqa",
            "medqa",
            "csqa",
            "recipegen",
            "summarization",
        ],
        help="Name of task to evaluate.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./",
        help="Path from where to read in the model outputs and reference answers to evaluate.",
    )
    parser.add_argument(
        "--output_path_annotator",
        type=str,
        default="./",
        help="Path where to store the outputs of the annotator evaluation model.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./",
        help="Path where the result scores will be stored.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path where the generated predictions will be stored.",
    )
    parser.add_argument(
        "--hf_token_path",
        type=str,
        default="",
        help="Path where the Hugging Face Llama token is stored.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for probability scaling during text generation.",
    )
    parser.add_argument(
        "--do_sample",
        default=False,
        action="store_true",
        help="Whether or not to activate sampling during text generation. Required for nucleus sampling.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Probability to exceed during top-p nucleus sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k tokens to consider for sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        help="Whether or not vLLM logprobs should be returned for vocabulary size.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum sequence length to be processed through vLLM.",
    )
    parser.add_argument(
        "--is_llama_3_chat_official",
        default=False,
        action="store_true",
        help=(
            "Whether or not we evaluate the official Llama 3 Chat model "
            "(requires different prompt formatting)."
        ),
    )

    return parser


def create_task_samples_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Hugging Face model name or path to evaluate.",
    )
    parser.add_argument(
        "--few_shot_path",
        type=str,
        help="Path where the few-shot samples are stored.",
    )
    parser.add_argument(
        "--corpus_samples_path",
        type=str,
        help="Path where the corpus samples are stored.",
    )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=3,
        help="Number of few-shots to include.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "bioqa",
            "medqa",
            "csqa",
            "recipegen",
            "summarization",
        ],
        help="Which task to generate task samples for.",
    )
    parser.add_argument(
        "--deduplication_ratio",
        type=int,
        default=85,
        help="Token set deduplication ratio for fuzzy deduplication. Number should be between 0 and 100.",
    )
    parser.add_argument(
        "--num_final_task_samples",
        type=int,
        help="Number of final task samples to sample from the created task samples (includes few-shots).",
    )
    parser.add_argument(
        "--output_path_raw",
        type=str,
        default="./",
        help="Path where the raw, unfiltered generated task samples will be stored.",
    )
    parser.add_argument(
        "--output_path_clean",
        type=str,
        default="./",
        help="Path where the clean, filtered generated task samples will be stored.",
    )
    parser.add_argument(
        "--output_path_error_msgs",
        type=str,
        default="./",
        help="Path where the indices and their format error messages will be stored.",
    )
    parser.add_argument(
        "--output_path_final",
        type=str,
        default="./",
        help="Path where the final task samples will be stored.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for probability scaling during text generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Probability to exceed during top-p nucleus sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k tokens to consider for sampling.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--max_tokenization_length",
        type=int,
        default=None,
        help="Maximum sequence length before truncation.",
    )

    return parser


def finetune_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Hugging Face model name or path to evaluate.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path where downloaded model is stored.",
    )
    parser.add_argument(
        "--task_samples_path",
        type=str,
        help="Path where the final, generated task samples to train on are stored. Choose `official` if you want to train on the official datasets.",
    )
    parser.add_argument(
        "--few_shots_only",
        default=False,
        action="store_true",
        help="Whether or not to only train on few-shots.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "bioqa",
            "medqa",
            "csqa",
            "recipegen",
            "summarization",
        ],
        help="Which task to generate task samples for.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Path where the finetuned LoRA and tokenizer will be stored.",
    )
    parser.add_argument(
        "--hf_token_path",
        type=str,
        default="",
        help="Path where the Hugging Face token is stored.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="Rank of the update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA scaling factor.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout rate of LoRA modules.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="Whether or not to include a bias term.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per device training batch size.",
    )
    parser.add_argument(
        "--use_bf16",
        default=False,
        action="store_true",
        help="Whether or not to use bfloat16 during training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate during training.",
    )
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs to run finetuning.",
    )
    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["epoch", "steps", "no"],
        help="At which points during training to log the progress.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Number of update steps between two logs.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="At which points during training to save the model.",
    )
    parser.add_argument(
        "--save_only_model",
        default=False,
        action="store_true",
        help="Whether or not to only save the model (adapters in this case), or also the optimizer and random states.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Which optimizer to use for finetuning.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="The warmup ratio to use for adjusting learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="The weight decay factor used during training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of workers to use for data loading.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="The number of checkpoints to save. If 1, only the most recent checkpoint will be kept.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Which type of learning rate scheduler to use.",
    )
    parser.add_argument(
        "--group_by_length",
        default=False,
        action="store_true",
        help="Whether or not to group the training samples in a batch by length.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="Transformers random seed.",
    )

    return parser
