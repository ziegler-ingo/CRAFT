import json
import gzip

import h5py

import torch


def get_c4_subfile_texts(c4_subfile):
    try:
        with gzip.open(c4_subfile, "rt", encoding="utf-8") as gzip_f:
            return [json.loads(f)["text"] for f in gzip_f]
    except (EOFError, ValueError, json.JSONDecodeError, TypeError):
        print(f"Found damaged file: {c4_subfile}.")
        return None


def get_jsonl_subfile_texts(subfile):
    try:
        with open(subfile, "r", encoding="utf-8") as jsonl_f:
            return [json.loads(f)["text"] for f in jsonl_f]
    except (EOFError, ValueError, json.JSONDecodeError, TypeError):
        print(f"Found damaged file: {subfile}.")
        return None


def retrieve_last_saved_uid(path, group_name):
    with h5py.File(path, "a") as h5_file:
        if group_name in h5_file:
            group = h5_file[group_name]
            uids = list(group.keys())  # type: ignore

            if uids:
                return uids[-1]
            else:
                return None
        else:
            group = h5_file.create_group(group_name)

            return None


def write_to_hdf5(path, group_name, uid, embeddings, compression, compression_opts):
    with h5py.File(path, "a") as h5_file:
        if group_name not in h5_file:
            raise ValueError("Group name not defined at time of writing to database.")
        else:
            group = h5_file[group_name]

        group.create_dataset(name=uid, data=embeddings, compression=compression, compression_opts=compression_opts)  # type: ignore


def embed_samples(samples, model, batch_size, dtype, to_cpu=False) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=dtype is torch.float16, dtype=dtype):  # type: ignore
        embeddings = model.encode(
            samples,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
    return embeddings if not to_cpu else embeddings.cpu()
