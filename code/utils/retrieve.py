import sys
import gzip
from collections import namedtuple

import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from sentence_transformers import util


TopPOutput = namedtuple(
    "TopPOutput",
    ["sim_metadata", "sim_values", "sim_idxs", "sim_values_mean", "sim_idxs_mean"],
)


def top_p_similarities(path, fs_embeddings, p, device):
    """
    Perform pre-filtering of a large H5 embedding database based on cosine similarity.
    This function retrieves the top-p percentile of most similar documents of a database split.
    """
    sim_metadata = []
    sim_values, sim_idxs = [], []
    sim_values_mean, sim_idxs_mean = [], []
    num_few_shots = fs_embeddings.shape[0]

    with h5py.File(path, "r") as h5_file:
        # "group" is the correct h5 database term, but we would call it dataset, e.g. C4
        for group_name, group in h5_file.items():
            print(f"Processing group: {group_name} of {list(h5_file.keys())}.")

            # "dataset" is the correct h5 database term, but we use it as e.g. C4 subfile
            # one "dataset" has shape [N, 384], N is number of samples (e.g. ~350k), 384 is embedding dim
            for dataset_name, dataset in tqdm(group.items(), file=sys.stdout):
                ds = torch.from_numpy(dataset[:]).to(device)  # shape [N, 384]
                n = ds.shape[0]

                sims = util.cos_sim(fs_embeddings, ds)
                sims_mean = sims.mean(dim=0, keepdim=True)

                # we want 50/50 sampling of top-1 similarity and top-num_few_shots-mean similarity
                # therefore, we divide by num_few_shots, and divide by 2 due to 50/50 split
                top_p_values, top_p_idxs = sims.topk(
                    int(p * n / num_few_shots / 2), dim=1
                )
                top_p_values_mean, top_p_idxs_mean = sims_mean.topk(
                    int(p * n / 2), dim=1
                )

                sim_metadata.append(
                    {
                        "group_name": group_name,
                        "dataset_name": dataset_name,
                        "length": top_p_idxs.shape[1],
                        "length_mean": top_p_idxs_mean.shape[1],
                    }
                )
                sim_values.append(top_p_values)
                sim_idxs.append(top_p_idxs)
                sim_values_mean.append(top_p_values_mean)
                sim_idxs_mean.append(top_p_idxs_mean)

    return TopPOutput(
        sim_metadata, sim_values, sim_idxs, sim_values_mean, sim_idxs_mean
    )


def top_k_indices(
    sim_metadata,
    sim_values,
    sim_idxs,
    sim_values_mean,
    sim_idxs_mean,
    k,
):
    """
    Return the top-k most similar indices per database subsplit.
    The indices of each subfile match the layout of the H5 embedding database.

    Returns
    _______
    idxs_per_subfile: list of lists of selected top indices per subfile
    """
    mixed_k = int(k / 2 * 1.2)  # +20% to accommodate potential duplicates
    num_few_shots = sim_values[0].shape[0]

    final_idxs = top_k_subgroup_idxs(
        values=sim_values,
        idxs=sim_idxs,
        k=mixed_k // num_few_shots,
        metadata=sim_metadata,
        subgroup="separate",
    )

    final_idxs_mean = top_k_subgroup_idxs(
        values=sim_values_mean,
        idxs=sim_idxs_mean,
        k=mixed_k,
        metadata=sim_metadata,
        subgroup="mean",
    )

    assert len(final_idxs) == len(
        final_idxs_mean
    ), "Final indices do not have same number of subfile lists."
    num_dedup_idxs = sum(len(l) for l in final_idxs)
    num_dedup_idxs_mean = sum(len(l) for l in final_idxs_mean)
    print(f"Number of deduplicated separate indices: {num_dedup_idxs}")
    print(f"Number of deduplicated mean indices: {num_dedup_idxs_mean}")
    print(f"Sum: {num_dedup_idxs + num_dedup_idxs_mean}")
    out = [sorted(list(set(f + fm))) for f, fm in zip(final_idxs, final_idxs_mean)]
    num_out = sum(len(l) for l in out)
    print(
        "Number of exact duplicates removed between separate and mean indices: "
        f"{num_dedup_idxs + num_dedup_idxs_mean - num_out}."
    )

    return out


def top_k_subgroup_idxs(values, idxs, k, metadata, subgroup):
    length_key = "length" if subgroup.startswith("separate") else "length_mean"

    values, idxs = torch.hstack(values), torch.hstack(idxs)
    _, top_k_idxs = values.topk(k=k, dim=1)
    filtered_idxs = mask_except(idxs, top_k_idxs, mask_value=-1)
    final_idxs = idxs_per_subfile(filtered_idxs, metadata, length_key)

    return final_idxs


def mask_except(tensor, except_idxs, mask_value):
    mask = torch.ones_like(tensor, dtype=torch.bool)
    mask.scatter_(1, except_idxs, False)
    masked = tensor.masked_fill(mask, mask_value)

    return masked


def idxs_per_subfile(indices, sim_metadata, length_key):
    idxs_per_subfile = []
    lengths = [sample[length_key] for sample in sim_metadata]

    for t in indices.split(lengths, dim=1):
        top_idxs = t[t != -1]

        if top_idxs.numel():
            # .unique() also sorts in ascending order
            idxs_per_subfile.append(top_idxs.unique().tolist())
        else:
            idxs_per_subfile.append([])

    return idxs_per_subfile


class JsonDataset(Dataset):
    def __init__(self, sim_metadata, indices_per_subfile, paths):
        self.sim_metadata = sim_metadata
        self.indices_per_subfile = indices_per_subfile
        self.paths = paths

    def __len__(self):
        return len(self.sim_metadata)

    def __getitem__(self, idx):
        sample = self.sim_metadata[idx]
        indices = self.indices_per_subfile[idx]
        path = self.paths[idx]

        docs = collect_docs(sample, indices, path)
        return docs


def build_paths(sim_metadata, args):
    """Store all paths in order of sim_metadata for parallel processing."""
    paths = [
        (
            args.c4_path
            if sample["group_name"] == "c4"
            else (
                args.stackexchange_path
                if sample["group_name"] == "stackexchange"
                else (
                    args.wikipedia_path
                    if sample["group_name"] == "wikipedia"
                    else args.wikihow_path
                )
            )
        )
        for sample in sim_metadata
    ]

    return paths


def collect_docs(sample, indices, path):
    if sample["group_name"] == "c4":
        path = f"{path}c4-{sample['dataset_name']}-of-01024.json.gz"
        docs = get_docs_from_json(path, indices)
    else:
        path = f"{path}{sample['dataset_name']}.jsonl"
        docs = get_docs_from_json(path, indices)
    return docs


def get_docs_from_json(path, indices):
    index_iterator = iter(indices)
    current_index = next(index_iterator, None)
    docs = []

    if current_index is None:
        return docs
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == current_index:
                    docs.append(line)
                    current_index = next(index_iterator, None)

                    if current_index is None:
                        break
    else:
        with open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == current_index:
                    docs.append(line)
                    current_index = next(index_iterator, None)

                    if current_index is None:
                        break
    return docs
