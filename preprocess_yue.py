#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, List

import ToJyutping
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from pebble import ProcessPool
from functools import partial

from tqdm import tqdm

TOKENIZER_NAME = "hon9kon9ize/bert-base-cantonese"
NUM_SHARDS = 10
MAX_WORKERS = 8
SHARDS_DIR = "./cantonese_processed_shards"
FINAL_DATA_FOLDER = "wikipedia_20231101.zh-yue.processed"
TOKEN_MAP_PATH = "yue_token_maps.json"

def cantonese_phonemize(example: Dict[str, Any],
                        tokenizer: AutoTokenizer,
                        text_column: str = "text") -> Dict[str, Any]:
    """
    ## Helper function that calls ToJyutping on a single row

    example: a single row from your dataset (which must have `text_column`).
    Returns the row plus 'jyutping' and 'input_ids'.
    """

    text = example[text_column]
    pairs = ToJyutping.get_jyutping_list(text)

    jyutping_list: List[str] = []
    for char, jyutping in pairs:
        if jyutping:
            jyutping_list.append(jyutping)
        else:
            # Handle punctuations
            jyutping_list.append(char)

    encoded = tokenizer(text, add_special_tokens=True, truncation=True)
    # input_ids is a list of token IDs, for example:
    # tokenizer("你今日食咗飯未")["input_ids"] -> [101, 872, 791, 3189, 7608, 1477, 7613, 3313, 102]
    input_ids = encoded["input_ids"]

    # Return new columns: "jyutping" and "input_ids"
    # You can store the entire list of romanizations or store them word-by-word
    # e.g. ["gam3", "ngaam1", "lou5", ...]
    example["jyutping"] = jyutping_list
    example["input_ids"] = input_ids

    # Optionally remove original text if you do not need it
    # But typically, you might remove it later with `remove_columns` in `.map(...)`.
    return example


def process_shard(i, dataset, root_directory, num_shards, tokenizer, text_column="text") -> None:
    """
    ## Main processing function for a shard (optional)

    Processes dataset shard i by phonemizing and tokenizing,
    then saves to disk at `root_directory/shard_{i}`.
    """

    directory = os.path.join(root_directory, f"shard_{i}")
    if os.path.exists(directory):
        print(f"[INFO] Shard {i} already exists! Skipping.")
        return

    print(f"[INFO] Processing shard {i} ...")
    shard = dataset.shard(num_shards=num_shards, index=i)

    # Apply the phonemization/tokenization map
    processed_shard = shard.map(
        lambda row: cantonese_phonemize(row, tokenizer, text_column),
        remove_columns=[text_column]  # if we want to remove original text
    )

    os.makedirs(directory, exist_ok=True)
    processed_shard.save_to_disk(directory)
    print(f"[INFO] Shard {i} saved to {directory}")


def build_and_save_token_map(dataset, token_map_path):
    """
    1. Gather all unique token IDs in the dataset (from 'input_ids').
    2. Build a new 'token_map' that might remap them (e.g., sorted enumeration).
    3. Save to JSON at token_map_path.
    """
    unique_tokens = set()
    for example in dataset:
        for tid in example["input_ids"]:
            unique_tokens.add(tid)

    # Sort them so it's reproducible
    unique_tokens = sorted(list(unique_tokens))

    # Build a mapping: old_id -> new_id
    # (If you want a 1-to-1 identity map, you can skip sorting or do something else.)
    token_map = {}
    for new_id, old_id in enumerate(unique_tokens):
        token_map[old_id] = new_id

    # Save to JSON
    with open(token_map_path, 'w', encoding='utf-8') as f:
        json.dump(token_map, f)

    print(f"[INFO] Built token map of size {len(token_map)}. Saved to {token_map_path}")


def main():
    dataset = load_dataset("wikimedia/wikipedia", "20231101.zh-yue")['train']
    print(f"[INFO] Loaded dataset size: {len(dataset) / 1000}k rows")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=False)

    # We define a partial to fix arguments except shard index
    process_shard_partial = partial(
        process_shard,
        dataset=dataset,
        root_directory=SHARDS_DIR,
        num_shards=NUM_SHARDS,
        tokenizer=tokenizer,
        text_column='text'
    )

    with ProcessPool(max_workers=MAX_WORKERS) as pool:
        pool.map(process_shard_partial, range(NUM_SHARDS), timeout=600)

    # Concatenate the shards
    shard_dirs = [
        d for d in os.listdir(SHARDS_DIR)
        if os.path.isdir(os.path.join(SHARDS_DIR, d))
    ]
    processed_datasets = []
    for sd in tqdm(shard_dirs):
        shard_path = os.path.join(SHARDS_DIR, sd)
        try:
            ds = load_from_disk(shard_path)
            processed_datasets.append(ds)
            print(f"[INFO] Shard loaded: {sd}")
        except Exception as e:
            print(f"[WARN] Failed to load {shard_path}: {e}")
            continue

    if not processed_datasets:
        raise ValueError("No shards loaded. Check if sharding or processing failed.")

    final_dataset = concatenate_datasets(processed_datasets)
    print(f"[INFO] Final dataset length: {len(final_dataset) / 1000}k rows")

    # Save final dataset
    os.makedirs(FINAL_DATA_FOLDER, exist_ok=True)
    final_dataset.save_to_disk(FINAL_DATA_FOLDER)
    print(f"[INFO] Dataset saved to {FINAL_DATA_FOLDER}")

    build_and_save_token_map(final_dataset, TOKEN_MAP_PATH)

if __name__ == "__main__":
    main()
