import os
from time import time

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from utils.embed import embed_samples
from utils.args import retrieve_docs_args
from utils.common import jsonl_generator
from utils import retrieve


parser = retrieve_docs_args()
args = parser.parse_args()

start = time()
device = args.device
model = SentenceTransformer(args.model_name).to(device)
TORCH_DTYPE = torch.float16 if args.use_amp_fp16 else torch.float32

# load all few_shot texts
few_shots = [fs for fs in jsonl_generator(args.few_shot_path, return_string=True)]
print("Starting embedding of few-shot samples...")
fs_embeddings = embed_samples(
    samples=few_shots, model=model, batch_size=32, dtype=TORCH_DTYPE, to_cpu=False
).to(TORCH_DTYPE)
print("Finished embedding few-shot samples.")

print("Starting calculation of top-p similarities...")
top_p_out = retrieve.top_p_similarities(
    path=args.database_path,
    fs_embeddings=fs_embeddings,
    p=args.top_p_percentile,
    device=args.device,
)
print("Finished calculation of top-p similarities.")

print("Starting calculation of top-k final document similarities...")
indices_per_subfile = retrieve.top_k_indices(
    sim_metadata=top_p_out.sim_metadata,
    sim_values=top_p_out.sim_values,
    sim_idxs=top_p_out.sim_idxs,
    sim_values_mean=top_p_out.sim_values_mean,
    sim_idxs_mean=top_p_out.sim_idxs_mean,
    k=args.num_samples_to_retrieve,
)
num_samples = sum(len(l) for l in indices_per_subfile)
print("Finished calculation of top-k final document similarities.")
print(f"Identified {num_samples} documents.")
paths = retrieve.build_paths(top_p_out.sim_metadata, args)
assert (
    len(top_p_out[0]) == len(indices_per_subfile) == len(paths)
), "Contents of metadata, indices per subfile, and paths do not match."

# retrieve final text documents from corpora based on selected top indices
print("Retrieving top-k documents from files...")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
most_similar_json = []
dataset = retrieve.JsonDataset(top_p_out.sim_metadata, indices_per_subfile, paths)
dataloader = DataLoader(dataset, batch_size=1, num_workers=2)
# open files and extract jsons from subfiles in parallel
for i, batch in enumerate(dataloader):
    most_similar_json.extend([t[0] for t in batch])
    print(f"Retrieved docs from file {i} of {len(top_p_out.sim_metadata)}.", end="\r")
end = time()
print("\nFinished retrieving top-k documents.")
print(f"Total retrieval took {(end-start)/60:.2f} minutes.")

print("Start saving retrieved documents...")
with open(args.saving_path, "w") as jsonl_file:
    for json_obj in most_similar_json:
        jsonl_file.write(json_obj)
print(f"Finished saving retrieved documents at {args.saving_path}.")
