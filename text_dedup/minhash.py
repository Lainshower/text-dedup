#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import glob
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
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
from text_dedup.utils import sha1_hash
from text_dedup.utils import xxh3_16hash
from text_dedup.utils import xxh3_32hash

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()
# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)
uf = UnionFind()
SIGNATURE_COLUMN = "__signatures__"


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

    Parameters
    ----------
    content : str
        The content to be embedded.
    idx : int
        The index of the content.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of n-grams.
    min_length : int
        The minimum length of the document in terms of tokens.
    hashranges : List[Tuple[int, int]]
        The ranges of hash values.
    permutations : np.ndarray
        The permutations for the minhash.
    hash_func : Callable
        The hash function to use.

    Returns
    -------
    Dict[str, Any]
        The hash values in each range and the index.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> max_hash = np.uint32((1 << 32) - 1)
    >>> modulo_prime = np.uint32((1 << 32) - 5)
    >>> PERMUTATIONS = (RNG.randint(1, modulo_prime, size=num_perm), RNG.randint(0, modulo_prime, size=num_perm))
    >>> res = embed_func(
    ...     content,
    ...     idx,
    ...     num_perm=num_perm,
    ...     ngram_size=ngram_size,
    ...     min_length=0,
    ...     hashranges=hashranges,
    ...     permutations=PERMUTATIONS,
    ...     hash_func=xxh3_32hash,
    ...     dtype=np.uint32,
    ...     max_hash=max_hash,
    ...     modulo_prime=modulo_prime,
    ... )
    >>> len(res[SIGNATURE_COLUMN])
    10
    >>> res[INDEX_COLUMN]
    0
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


def load_jsonl_files(input_dir: str, num_proc: int) -> List[Dict]:
    """Load all jsonl files from directory recursively using multiprocessing."""
    files = glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True)
    logger.info(f"Found {len(files)} .jsonl files in {input_dir}")
    
    with mp.Pool(num_proc) as pool:
        all_data = list(tqdm(
            pool.imap(process_file, files),
            total=len(files),
            desc="Loading JSONL files"
        ))
    
    return [item for sublist in all_data for item in sublist]


def save_jsonl(data: List[Dict], output_path: str):
    """Save data to jsonl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(data)} documents to {output_path}")


@click.command
@click.option(
    "--input_dir",
    type=str,
    required=True,
    help="Input directory containing JSONL files",
)
@click.option(
    "--output",
    type=str,
    required=True,
    help="Output path for deduplicated data",
)
@click.option(
    "--batch_size",
    type=int,
    default=10000,
    help="Batch size for processing",
)
@click.option(
    "--num_proc",
    type=int,
    default=mp.cpu_count(),
    help="Number of processes to use",
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
    input_dir: str,
    output: str,
    batch_size: int,
    num_proc: int,
    threshold: float,
    num_perm: int,
    ngram: int,
    min_length: int,
    hash_bits: int,
    b: Optional[int],
    r: Optional[int],
):
    global uf
    uf.reset()
    timer = Timer()

    # 64 bit config is backwards compatibility mode.
    # it uses 64 bit types but almost entirely 32bit data, except for one mersenne prime 2^61
    # why legacy implementations used mersenne primes for modulo:
    # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16 bit config does not use a mersenne prime.
        # The original reason for using mersenne prime was speed.
        # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    # defaults to backwards compatible HASH_BITS = 64, which is np.uint64 dtypes with 32bit hashes
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(hash_bits, HASH_CONFIG[64])

    match hash_bits:
        case 16:
            hash_func = xxh3_16hash
        case _:
            hash_func = xxh3_32hash

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
    
    logger.info(f"Using LSH parameters: B={B}, R={R}")
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
            # Load local JSONL files using multiprocessing
            data = load_jsonl_files(input_dir, num_proc)
            ds = datasets.Dataset.from_list(data)
            
            # Add index column if not present
            if INDEX_COLUMN not in ds.column_names:
                ds = ds.add_column(INDEX_COLUMN, range(len(ds)))
            
            # Filter by minimum length
            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x['text'].lower())) >= min_length,  # Always use 'text' field
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
                input_columns=['text', INDEX_COLUMN],  # Always use 'text' field
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                num_proc=num_proc,
                with_indices=False,
                desc="Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            NUM_SHARDS = np.ceil(LEN_EMBEDDED / batch_size).astype(int)

        with timer("Clustering"):
            edges = []
            
            for i in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating MinHashes...",
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=batch_size,
                )
                for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            logger.info(f"Number of clusters: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)
            logger.info(f"Number of edges: {len(set(edges))}")

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            
            # Keep only one instance per cluster while preserving all metadata
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
                num_proc=num_proc,
                desc="Filtering clusters...",
            )

        with timer("Saving"):
            # Convert to list of dicts for JSONL saving
            final_data = final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])
            output_data = final_data.to_list()
            
            # Save as JSONL
            if output.endswith('.jsonl'):
                save_jsonl(output_data, output)
            else:
                save_jsonl(output_data, output + '.jsonl')

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=no-value-for-parameter
    main()