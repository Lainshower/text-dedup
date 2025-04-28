import os
import glob
import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import re
import datasets
from text_dedup.utils import ngrams, optimal_param, xxh3_16hash, xxh3_32hash

NON_ALPHA = re.compile(r"\W", re.UNICODE)
SEED = 42
RNG = np.random.RandomState(SEED)

# --- MinHash dedup core logic (single node, multi-proc) ---
def process_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if 'text' in item:
                    data.append(item)
            except Exception:
                continue
    return data

def embed_func(content, idx, num_perm, ngram_size, min_length, hashranges, permutations, hash_func, dtype, max_hash, modulo_prime):
    a, b = permutations
    tokens = {
        bytes(" ".join(t).lower(), "utf-8") for t in ngrams(NON_ALPHA.split(content.lower()), ngram_size, min_length)
    }
    hashvalues = np.array([hash_func(token) for token in tokens], dtype=dtype).reshape(len(tokens), 1)
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    masks = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {"__signatures__": Hs, "__index__": idx}

def dedup_records(records, column, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r, out_path):
    # Prepare dataset
    import datasets
    ds = datasets.Dataset.from_list(records)
    if "__index__" not in ds.column_names:
        ds = ds.add_column("__index__", range(len(ds)))
    ds = ds.filter(
        lambda x: len(NON_ALPHA.split(x[column].lower())) >= min_length,
        num_proc=num_proc,
    )
    LEN_DATASET = len(ds)
    # Hash config
    HASH_CONFIG = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(hash_bits, HASH_CONFIG[64])
    hash_func = xxh3_32hash if hash_bits != 16 else xxh3_16hash
    if b is not None and r is not None:
        B, R = b, r
    else:
        B, R = optimal_param(
            threshold,
            num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = (
        RNG.randint(1, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),
        RNG.randint(0, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),
    )
    # MinHashing
    embedded = ds.map(
        function=lambda content, idx: embed_func(
            content, idx, num_perm, ngram, min_length, HASH_RANGES, PERMUTATIONS, hash_func, DTYPE, MAX_HASH, MODULO_PRIME
        ),
        input_columns=[column, "__index__"],
        remove_columns=[col for col in ds.column_names if col != "__index__"],
        num_proc=num_proc,
        with_indices=False,
        desc=f"Fingerprinting...",
    )
    LEN_EMBEDDED = len(embedded)
    NUM_SHARDS = 1  # Always process all data at once
    # Clustering
    from text_dedup.utils import UnionFind
    uf = UnionFind()
    SIGNATURE_COLUMN = "__signatures__"
    INDEX_COLUMN = "__index__"
    HASH_TABLES = [defaultdict(set) for _ in range(B)]
    edges = []
    for i in tqdm(range(0, NUM_SHARDS), dynamic_ncols=True, desc=f"Iterating MinHashes..."):
        embedded_shard = embedded.shard(
            num_shards=NUM_SHARDS,
            index=i,
            contiguous=True,
            writer_batch_size=LEN_EMBEDDED,
        )
        for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
            for j, H in enumerate(Hs):
                HASH_TABLES[j][H].add(key)
    for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc=f"Clustering..."):
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                edges.append((x, idx))
                uf.union(x, idx)
    # Filtering
    ds = ds.map(
        function=lambda record: {"__cluster__": uf.find(record[INDEX_COLUMN])},
        with_indices=False,
        num_proc=num_proc,
        new_fingerprint=str(np.random.getrandbits(128)),
        desc=f"Finding clusters...",
    )
    final_data = ds.filter(
        function=lambda record: record["__cluster__"] == record[INDEX_COLUMN],
        with_indices=False,
        num_proc=num_proc,
        desc=f"Filtering clusters...",
    )
    # Save
    final_data = final_data.remove_columns(["__cluster__", INDEX_COLUMN])
    output_data = final_data.to_list()
    with open(out_path, 'w', encoding='utf-8') as fout:
        for item in output_data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Deduped {LEN_DATASET} -> {len(final_data)} records. Saved to {out_path}")

# --- Merge and dedup main ---
def merge_language_outputs_and_dedup(output_dir, column, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r):
    merged_count = 0
    for lang_dir in sorted(os.listdir(output_dir)):
        lang_path = os.path.join(output_dir, lang_dir)
        if not os.path.isdir(lang_path):
            continue
        dedup_files = sorted(glob.glob(os.path.join(lang_path, 'dedup_node*.jsonl')))
        if not dedup_files:
            continue
        out_path = os.path.join(output_dir, f"{lang_dir}.jsonl")
        print(f"Merging {len(dedup_files)} files for language '{lang_dir}' into {out_path}")
        with open(out_path, 'w', encoding='utf-8') as fout:
            for dedup_file in dedup_files:
                with open(dedup_file, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)
        merged_count += 1
        # --- After merging, run dedup on the merged file ---
        print(f"Deduplicating {out_path} ...")
        records = process_jsonl_file(out_path)
        final_out_path = os.path.join(output_dir, f"{lang_dir}.final.jsonl")
        dedup_records(records, column, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r, final_out_path)
    print(f"Merged and deduped {merged_count} languages.")

def main():
    parser = argparse.ArgumentParser(description="Merge dedup_node*.jsonl files per language into a single <lang>.jsonl file, then deduplicate again using minhash.")
    parser.add_argument('output_dir', type=str, help='Path to the dedup output directory (containing language subdirs)')
    parser.add_argument('--column', type=str, default='text', help='Column name containing text')
    parser.add_argument('--num_proc', type=int, default=8, help='Number of processes to use')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold')
    parser.add_argument('--num_perm', type=int, default=256, help='Number of permutations')
    parser.add_argument('--ngram', type=int, default=3, help='N-gram size')
    parser.add_argument('--min_length', type=int, default=20, help='Minimum document length')
    parser.add_argument('--hash_bits', type=int, default=64, help='Number of hash bits')
    parser.add_argument('--b', type=int, default=None, help='Number of bands (if None, will be optimized)')
    parser.add_argument('--r', type=int, default=None, help='Number of rows (if None, will be optimized)')
    args = parser.parse_args()
    merge_language_outputs_and_dedup(
        args.output_dir, args.column, args.num_proc, args.threshold, args.num_perm, args.ngram, args.min_length, args.hash_bits, args.b, args.r
    )

if __name__ == "__main__":
    main() 