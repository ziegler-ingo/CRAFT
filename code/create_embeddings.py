import os
import sys

from tqdm import tqdm
import numpy as np

import torch
from sentence_transformers import SentenceTransformer

from utils.args import create_embeddings_args
from utils import embed


parser = create_embeddings_args()
args = parser.parse_args()


TORCH_DTYPE = torch.float16 if args.use_amp_fp16 else torch.float32
NUMPY_DTYPE = np.float16 if args.use_amp_fp16 else np.float32
MAX_C4_SUBFILE_IDX = 1023
MAX_WIKIPEDIA_SUBFILE_IDX = 12
MAX_STACKEXCHANGE_SUBFILE_IDX = 13

device = args.device
model = SentenceTransformer(args.model_name).to(device)
db_path = args.saving_path + args.saving_file

if not os.path.exists(args.saving_path):
    os.makedirs(args.saving_path)
    print(f"Created directory path {args.saving_path}")

if args.group_name == "c4":
    c4_subfiles = [
        f"{args.data_path}c4-train.{i:05d}-of-01024.json.gz"
        for i in range(MAX_C4_SUBFILE_IDX + 1)
    ]
    last_uid = embed.retrieve_last_saved_uid(db_path, args.group_name)

    # define next starting point from already processed files in the database
    if last_uid is not None:
        _, last_subfile_idx_text = last_uid.split(".")
        last_subfile_idx = int(last_subfile_idx_text)

        if last_subfile_idx == MAX_C4_SUBFILE_IDX:
            print("All C4 subfiles have been processed.")
            sys.exit(0)
        else:
            next_subfile_idx = last_subfile_idx + 1
    else:
        next_subfile_idx = 0

    # start processing
    for c4_subfile in tqdm(c4_subfiles[next_subfile_idx:], file=sys.stdout):
        uid = c4_subfile.split("/")[-1].split("-")[1]  # uid = "train.xxxxx"
        print(f"Embedding file {uid}.")
        samples = embed.get_c4_subfile_texts(c4_subfile)
        if samples is not None:
            embeddings = embed.embed_samples(
                samples=samples,
                model=model,
                batch_size=args.batch_size,
                dtype=TORCH_DTYPE,
                to_cpu=True,
            )
            embeddings = np.array(embeddings, dtype=NUMPY_DTYPE)
            embed.write_to_hdf5(
                path=db_path,
                group_name=args.group_name,
                uid=uid,
                embeddings=embeddings,
                compression=args.compression,
                compression_opts=args.compression_opts,
            )
elif args.group_name == "wikipedia":
    wikipedia_subfiles = [
        f"{args.data_path}wiki_{i:02d}.jsonl"
        for i in range(MAX_WIKIPEDIA_SUBFILE_IDX + 1)
    ]
    last_uid = embed.retrieve_last_saved_uid(db_path, args.group_name)

    # define next starting point from already processed files in the database
    if last_uid is not None:
        _, last_subfile_idx_text = last_uid.split("_")
        last_subfile_idx = int(last_subfile_idx_text)

        if last_subfile_idx == MAX_WIKIPEDIA_SUBFILE_IDX:
            print("All Wikipedia subfiles have been processed.")
            sys.exit(0)
        else:
            next_subfile_idx = last_subfile_idx + 1
    else:
        next_subfile_idx = 0

    # start processing
    for wikipedia_subfile in tqdm(
        wikipedia_subfiles[next_subfile_idx:], file=sys.stdout
    ):
        uid = wikipedia_subfile.split("/")[-1].split(".")[0]  # uid = "wiki_xx"
        print(f"Embedding file {uid}.")
        samples = embed.get_jsonl_subfile_texts(wikipedia_subfile)
        if samples is not None:
            embeddings = embed.embed_samples(
                samples=samples,
                model=model,
                batch_size=args.batch_size,
                dtype=TORCH_DTYPE,
                to_cpu=True,
            )
            embeddings = np.array(embeddings, dtype=NUMPY_DTYPE)
            embed.write_to_hdf5(
                path=db_path,
                group_name=args.group_name,
                uid=uid,
                embeddings=embeddings,
                compression=args.compression,
                compression_opts=args.compression_opts,
            )
elif args.group_name == "wikihow":
    # there is only one file for wikihow
    last_uid = embed.retrieve_last_saved_uid(db_path, args.group_name)

    if last_uid is not None:
        print("Wikihow has been processed.")
        sys.exit(0)
    else:
        next_subfile_idx = 0

    # start processing
    uid = args.data_path.split("/")[-1].split(".")[0]  # uid = "train"
    print(f"Embedding file {uid}.")
    samples = embed.get_jsonl_subfile_texts(args.data_path)
    if samples is not None:
        embeddings = embed.embed_samples(
            samples=samples,
            model=model,
            batch_size=args.batch_size,
            dtype=TORCH_DTYPE,
            to_cpu=True,
        )
        embeddings = np.array(embeddings, dtype=NUMPY_DTYPE)
        embed.write_to_hdf5(
            path=db_path,
            group_name=args.group_name,
            uid=uid,
            embeddings=embeddings,
            compression=args.compression,
            compression_opts=args.compression_opts,
        )
elif args.group_name == "stackexchange":
    stackexchange_subfiles = [
        f"{args.data_path}stack_{i:02d}.jsonl"
        for i in range(MAX_STACKEXCHANGE_SUBFILE_IDX + 1)
    ]
    last_uid = embed.retrieve_last_saved_uid(db_path, args.group_name)

    # define next starting point from already processed files in the database
    if last_uid is not None:
        _, last_subfile_idx_text = last_uid.split("_")
        last_subfile_idx = int(last_subfile_idx_text)

        if last_subfile_idx == MAX_STACKEXCHANGE_SUBFILE_IDX:
            print("All Stackexchange subfiles have been processed.")
            sys.exit(0)
        else:
            next_subfile_idx = last_subfile_idx + 1
    else:
        next_subfile_idx = 0

    # start processing
    for stackexchange_subfile in tqdm(
        stackexchange_subfiles[next_subfile_idx:], file=sys.stdout
    ):
        uid = stackexchange_subfile.split("/")[-1].split(".")[0]  # uid = "stack_xx"
        print(f"Embedding file {uid}.")
        samples = embed.get_jsonl_subfile_texts(stackexchange_subfile)
        if samples is not None:
            embeddings = embed.embed_samples(
                samples=samples,
                model=model,
                batch_size=args.batch_size,
                dtype=TORCH_DTYPE,
                to_cpu=True,
            )
            embeddings = np.array(embeddings, dtype=NUMPY_DTYPE)
            embed.write_to_hdf5(
                path=db_path,
                group_name=args.group_name,
                uid=uid,
                embeddings=embeddings,
                compression=args.compression,
                compression_opts=args.compression_opts,
            )
