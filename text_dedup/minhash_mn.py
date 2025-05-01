#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com) + Multi-node support
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import glob
import json
import socket
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import click
import datasets
import numpy as np
from tqdm import tqdm

from text_dedup import logger
from text_dedup.utils import CLUSTER_COLUMN
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import ngrams
from text_dedup.utils import optimal_param
from text_dedup.utils import xxh3_16hash
from text_dedup.utils import xxh3_32hash

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()
mp.set_start_method("fork", force=True)
uf = UnionFind()
SIGNATURE_COLUMN = "__signatures__"

def get_node_info() -> Tuple[int, int]:
    """Get node ID and total number of nodes from hostname."""
    hostname = socket.gethostname()
    try:
        # Assuming hostname ends with node number (e.g., 'server-01')
        node_id = int(hostname[-2:])
        # Get total nodes from environment variable or default to 1
        total_nodes = int(os.getenv('TOTAL_NODES', '1'))
        return node_id, total_nodes
    except ValueError:
        logger.warning(f"Could not parse node ID from hostname {hostname}, defaulting to single node mode")
        return 0, 1

def distribute_files(files: List[str], node_id: int, total_nodes: int) -> List[str]:
    """
    return [f for i, f in enumerate(files) if i % total_nodes == node_id]
    """
    """Distribute files among nodes in contiguous blocks (not round-robin)."""
    n = len(files)
    chunk_size = (n + total_nodes - 1) // total_nodes  # ceil division
    start = node_id * chunk_size
    end = min(start + chunk_size, n)
    return files[start:end]

def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    hash_func: Callable,
    dtype: type,
    max_hash: np.uint,
    modulo_prime: np.uint,
) -> dict[str, Any]:
    """
    Calculate hash values for the content.
    [Original docstring preserved]
    """
    a, b = permutations
    tokens: set[bytes] = {
        bytes(" ".join(t).lower(), "utf-8") for t in ngrams(NON_ALPHA.split(content.lower()), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array([hash_func(token) for token in tokens], dtype=dtype).reshape(len(tokens), 1)
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    masks: np.ndarray = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    Hs: list[bytes] = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return {SIGNATURE_COLUMN: Hs, INDEX_COLUMN: idx}

def process_file(file_path: str) -> List[Dict]:
    """Process a single JSONL file and return its contents."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    if 'text' not in item:
                        logger.warning(f"Skipping line {line_num} in {file_path}: 'text' field not found")
                        continue
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {line_num} in {file_path}")
                    continue
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
    return data

def load_jsonl_files(input_dir: str, num_proc: int, node_id: int, total_nodes: int) -> List[Dict]:
    """Load JSONL files distributed among nodes using multiprocessing."""
    all_files = glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True)
    node_files = distribute_files(all_files, node_id, total_nodes)
    logger.info(f"Node {node_id}/{total_nodes} processing {len(node_files)} out of {len(all_files)} files")
    
    with mp.Pool(num_proc) as pool:
        all_data = list(tqdm(
            pool.imap(process_file, node_files),
            total=len(node_files),
            desc=f"Node {node_id}: Loading JSONL files"
        ))
    
    return [item for sublist in all_data for item in sublist]

def save_jsonl(data: List[Dict], output_path: str, node_id: int):
    """Save data to node-specific JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    node_output = f"{os.path.splitext(output_path)[0]}_node{node_id:02d}.jsonl"
    with open(node_output, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Node {node_id}: Saved {len(data)} documents to {node_output}")

def dedup_from_filelist(file_list, output_path, column, batch_size, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r, node_id, total_nodes):
    global uf
    uf.reset()
    timer = Timer()

    # Hash configuration stays the same
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(hash_bits, HASH_CONFIG[64])
    hash_func = xxh3_32hash if hash_bits != 16 else xxh3_16hash

    # Compute optimal LSH parameters if not provided
    if b is not None and r is not None:
        B, R = b, r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
        # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
        # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
        # Both effects move the result towards more documents being considered duplicates.
        B, R = optimal_param(
            threshold,
            num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )
    logger.info(f"Node {node_id}: Using LSH parameters: B={B}, R={R}")
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES = [defaultdict(set) for _ in range(B)]
    # for minhash, we need to make a lot of hashes(=num_perms).
    # In many previous implementations, this is achieved through a method described in
    # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
    # There we start with a know good hash x (=hash_func) and permutate it as the following:
    # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
    # the following produces these a, b pairs
    PERMUTATIONS = (
        RNG.randint(1, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),  # a is a multiplier so should not be 0
        RNG.randint(0, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),  # b
    )
    with timer("Total"):
        with timer("Loading"):
            logger.info(f"Node {node_id}: Assigned {len(file_list)} files for deduplication")
            with mp.Pool(num_proc) as pool:
                all_data = list(tqdm(
                    pool.imap(process_file, file_list),
                    total=len(file_list),
                    desc=f"Node {node_id}: Loading JSONL files"
                ))
            data = [item for sublist in all_data for item in sublist]
            ds = datasets.Dataset.from_list(data)
            # Add index column if not present
            if INDEX_COLUMN not in ds.column_names:
                ds = ds.add_column(INDEX_COLUMN, range(len(ds)))
            # Filter by minimum length
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[column].lower())) >= min_length,
                num_proc=num_proc,
            )
        LEN_DATASET = len(ds)
        with timer("MinHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": ngram,
                    "min_length": min_length,
                    "permutations": PERMUTATIONS,
                    "hash_func": hash_func,
                    "dtype": DTYPE,
                    "max_hash": MAX_HASH,
                    "modulo_prime": MODULO_PRIME,
                },
                input_columns=[column, INDEX_COLUMN],
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                num_proc=num_proc,
                with_indices=False,
                desc=f"Node {node_id}: Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            if batch_size is None:
                NUM_SHARDS = 1
            else:
                NUM_SHARDS = np.ceil(LEN_EMBEDDED / batch_size).astype(int)
        with timer("Clustering"):
            edges = []
            for i in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc=f"Node {node_id}: Iterating MinHashes...",
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=batch_size if batch_size is not None else LEN_EMBEDDED,
                )
                for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
                    for j, H in enumerate(Hs):
                        HASH_TABLES[j][H].add(key)
            logger.info(f"Node {node_id}: Number of clusters: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc=f"Node {node_id}: Clustering..."):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)
            logger.info(f"Node {node_id}: Number of edges: {len(set(edges))}")
        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc=f"Node {node_id}: Finding clusters...",
            )
            # Keep only one instance per cluster while preserving all metadata
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
                num_proc=num_proc,
                desc=f"Node {node_id}: Filtering clusters...",
            )
        with timer("Saving"):
            # Convert to list of dicts for JSONL saving
            columns_to_remove = [c for c in [CLUSTER_COLUMN, INDEX_COLUMN] if c in final_data.column_names]
            if columns_to_remove:
                final_data = final_data.remove_columns(columns_to_remove)
            output_data = final_data.to_list()
            save_jsonl(output_data, output_path, node_id)
    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"Node {node_id}: Before:{LEN_DATASET}")
    logger.info(f"Node {node_id}: After:{len(final_data)}")

@click.command()
@click.option(
    "--input_dirs",
    type=str,
    default=None,
    help="Space or comma separated list of input directories containing language folders",
)
@click.option(
    "--input_dir",
    type=str,
    default=None,
    help="(Deprecated) Single input directory containing JSONL files",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output path for deduplicated data (directory)",
)
@click.option(
    "--column",
    type=str,
    default="text",
    help="Column name containing text",
)
@click.option(
    "--batch_size",
    type=int,
    help="Batch size for processing",
)
@click.option(
    "--num_proc",
    type=int,
    default=mp.cpu_count(),
    help="Number of processes to use per node",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    help="Similarity threshold",
)
@click.option(
    "--num_perm",
    type=int,
    default=256,
    help="Number of permutations",
)
@click.option(
    "--ngram",
    type=int,
    default=3,
    help="N-gram size",
)
@click.option(
    "--min_length",
    type=int,
    default=5,
    help="Minimum document length",
)
@click.option(
    "--hash_bits",
    type=int,
    default=64,
    help="Number of hash bits",
)
@click.option(
    "--b",
    type=int,
    default=None,
    help="Number of bands (if None, will be optimized)",
)
@click.option(
    "--r",
    type=int,
    default=None,
    help="Number of rows (if None, will be optimized)",
)
def main(
    input_dirs,
    input_dir,
    output,
    column,
    batch_size,
    num_proc,
    threshold,
    num_perm,
    ngram,
    min_length,
    hash_bits,
    b,
    r,
):
    node_id, total_nodes = get_node_info()
    logger.info(f"Running on node {node_id} of {total_nodes} total nodes")
    if input_dirs:
        dirs = [d for d in input_dirs.replace(',', ' ').split() if d]
        lang_files = defaultdict(list)
        for d in dirs:
            for lang in os.listdir(d):
                lang_dir = os.path.join(d, lang)
                if os.path.isdir(lang_dir):
                    files = glob.glob(os.path.join(lang_dir, '*.jsonl'))
                    lang_files[lang].extend(files)
        logger.info(f"Node {node_id}: Found languages: {list(lang_files.keys())}")
        for lang, files in lang_files.items():
            logger.info(f"Node {node_id}: Processing language {lang} with {len(files)} files")
            if not files:
                continue
            lang_output_dir = os.path.join(output, lang)
            os.makedirs(lang_output_dir, exist_ok=True)
            lang_output_path = os.path.join(lang_output_dir, f"dedup_node{node_id:02d}.jsonl")
            dedup_from_filelist(
                files, lang_output_path, column, batch_size, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r, node_id, total_nodes
            )
        return
    # Fallback: original single-dir logic
    if input_dir:
        all_files = glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True)
        dedup_from_filelist(
            all_files, output, column, batch_size, num_proc, threshold, num_perm, ngram, min_length, hash_bits, b, r, node_id, total_nodes
        )

if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()
