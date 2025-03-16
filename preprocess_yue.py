#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
from typing import Dict, Any

import ToJyutping
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
from transformers import AutoTokenizer

from pebble import ProcessPool
from functools import partial

from tqdm import tqdm


PUNCTUATIONS = [
    '，',
    '。',
    '！',
    '？',
]

def cantonese_phonemize(example: Dict[str, Any],
                        tokenizer,
                        text_column: str = "text") -> Dict[str, Any]:
    """
    ## Helper function that calls ToJyutping on a single row

    example: a single row from your dataset (which must have `text_column`).
    Returns the row plus 'jyutping' and 'input_ids'.
    """

    text = example[text_column]
    # Step A: Convert text to a list of (char, Jyutping) pairs
    pairs = ToJyutping.get_jyutping_list(text)

    # Step B: Collect the romanizations (ignore punctuation or None)
    # e.g. [("咁","gam3"), ("啱","ngaam1"), ... , (",", None)]
    jyutping_list = []
    for char, jyutping in pairs:
        if jyutping:
            jyutping_list.append(jyutping)
        else:
            # Handle punctuations
            jyutping_list.append(char)

    # Step C: For a subword tokenizer, we can get input_ids from the original text
    # or from some processed version. For simplicity, let's do original text:
    encoded = tokenizer(text, add_special_tokens=True, truncation=True)
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


def main():
    # -- A. Load config
    config_path = "Configs/config_cantonese.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # -- B. Load your dataset
    # Replace with your actual dataset. e.g.:
    #   dataset = load_dataset("text", data_files={"train": "my_cantonese_texts.txt"})["train"]
    # Or from HF: dataset = load_dataset("your-hf-cantonese-dataset")["train"]
    dataset = load_dataset("wikimedia/wikipedia", "20231101.zh-yue")['train']
    print("[INFO] Loaded dataset size:", len(dataset))

    # The text column name might not be "text"; adapt if your dataset uses another field
    text_column = config["dataset_params"].get("text_column", "text")

    # -- C. Initialize the tokenizer
    # This could be any suitable tokenizer for your text, or a new one trained for Cantonese.
    # For demonstration, let's use "bert-base-chinese" or a relevant Cantonese model
    tokenizer_name = config["dataset_params"]["tokenizer"]  # e.g. "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)

    # -- D. Sharding / Multiprocessing
    root_directory = "./cantonese_processed_shards"
    num_shards = config["dataset_params"].get("num_shards", 10)  # adapt as needed
    max_workers = config["dataset_params"].get("max_workers", 4)

    # We define a partial to fix arguments except shard index
    process_shard_partial = partial(
        process_shard,
        dataset=dataset,
        root_directory=root_directory,
        num_shards=num_shards,
        tokenizer=tokenizer,
        text_column=text_column
    )

    # Use a process pool for concurrency
    # or comment out if you want single-threaded
    with ProcessPool(max_workers=max_workers) as pool:
        pool.map(process_shard_partial, range(num_shards), timeout=600)

    # -- E. Concatenate the shards
    from datasets import load_from_disk, concatenate_datasets
    shard_dirs = [
        d for d in os.listdir(root_directory)
        if os.path.isdir(os.path.join(root_directory, d))
    ]
    processed_datasets = []
    for sd in shard_dirs:
        shard_path = os.path.join(root_directory, sd)
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
    print("[INFO] Final dataset length:", len(final_dataset))

    # -- F. Save final dataset
    final_data_folder = config["data_folder"]  # e.g. "cantonese_dataset_processed"
    os.makedirs(final_data_folder, exist_ok=True)
    final_dataset.save_to_disk(final_data_folder)
    print("[INFO] Dataset saved to", final_data_folder)

    # -- G. (Optional) Prune tokens or do additional steps
    # For example, gather unique token IDs in 'input_ids' if you plan on pruning,
    # or build a new token map for lowercasing, etc. The logic is similar to your
    # English script.

    # If you want to replicate the “token pruning” approach exactly:
    # 1. Load the final dataset with a simple loader that yields `input_ids`.
    # 2. Collect unique IDs, do any case or special token handling.
    # 3. Save out a `token_maps.json`.

    print("[INFO] Preprocessing complete.")


if __name__ == "__main__":
    main()
