
import os
import yaml
import json
from datasets import load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
from pebble import ProcessPool
from phonemize import phonemize
import phonemizer
from transformers import AutoTokenizer
from simple_loader import FilePathDataset, build_dataloader
from dataloader import build_dataloader as build_dataloader_from_dataloader
from functools import partial

NUM_SHARDS = 2
MAX_WORKERS = 8

def process_shard(i, dataset, root_directory, num_shards, global_phonemizer, tokenizer) -> None:
    directory = root_directory + "/shard_" + str(i)
    if os.path.exists(directory):
        print("Shard %d already exists!" % i)
        return
    
    print("Processing shard %d ..." % i)
    shard = dataset.shard(num_shards=num_shards, index=i)
    # A HF wikipedia dataset has id, url, title, text columns.

    processed_dataset = shard.map(lambda t: phonemize(t['text'], global_phonemizer, tokenizer), remove_columns=['text'])
    
    # Save to local disk
    if not os.path.exists(directory):
        os.makedirs(directory)
    processed_dataset.save_to_disk(directory)

def main():
    config_path = "Configs/config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize phonemizer and tokenizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', preserve_punctuation=True, with_stress=True)
    os.environ["TRUST_REMOTE_CODE"] = "True"
    tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'], use_fast=False)

    # Process dataset
    dataset = load_dataset("wikipedia", "20220301.en")['train'].select(range(2))  # load only first 2000 rows for small experiment
    root_directory = "./wiki_phoneme"  # set up root directory for multiprocessor processing
    

    process_shard_partial = partial(
        process_shard,
        dataset=dataset,
        root_directory=root_directory,
        num_shards=NUM_SHARDS,
        global_phonemizer=global_phonemizer,
        tokenizer=tokenizer
    )

    # change this to the number of CPU cores your machine has 
    with ProcessPool(max_workers=MAX_WORKERS) as pool:
        pool.map(process_shard_partial, range(NUM_SHARDS), timeout=600)

    # Collect all shards to form the processed dataset
    output = [dI for dI in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, dI))]
    datasets_list = []
    for o in output:
        directory = root_directory + "/" + o
        try:
            shard = load_from_disk(directory)
            datasets_list.append(shard)
            print("%s loaded" % o)
        except Exception:
            continue
    if not datasets_list:
        raise ValueError("No dataset shards were loaded. Please ensure that all shards are processed.")
    
    dataset = concatenate_datasets(datasets_list)
    dataset.save_to_disk(config['data_folder'])
    print("Dataset saved to %s" % config['data_folder'])
    print("Dataset size:", dataset)

    # Remove unnecessary tokens from the pre-trained tokenizer
    file_data = FilePathDataset(dataset)
    loader = build_dataloader(file_data, num_workers=16, batch_size=128)
    # When decoded, this 3039 token is <formula>
    special_token = config['dataset_params']['word_separator']

    # 
    unique_index = set([special_token])  # Initialize as a set
    for batch in tqdm(loader):
        unique_index.update(batch)  # Use update to add multiple elements at once
                                                                                                                                                                                            
    lower_tokens = set()
    for t in tqdm(unique_index):
        word = tokenizer.decode([t])
        lower_word = word.lower()
        if lower_word != word:
            t_new = tokenizer.encode([lower_word])[0]
            lower_tokens.add(t_new)
        else:
            lower_tokens.add(t)

                                                                                                                                                                                                                            
    token_maps = {}
    lower_tokens = sorted(list(lower_tokens))
    # each entry is `token_id: {'word': word, 'token': token_id}`
    print("Lower-cased tokens:", len(lower_tokens))
    for t in tqdm(unique_index):
        word = tokenizer.decode([t]).lower()
        new_t = tokenizer.encode([word])[0]
        token_maps[t] = {'word': word, 'token': lower_tokens.index(new_t)}

    with open(config['dataset_params']['token_maps'], 'w') as handle:
        json.dump(token_maps, handle)
    print("Token mapper saved to %s" % config['dataset_params']['token_maps'])

    # Test the dataset with dataloader
    train_loader = build_dataloader_from_dataloader(dataset, batch_size=32, num_workers=0, dataset_config=config['dataset_params'])
    _, (words, labels, phonemes, input_lengths, masked_indices) = next(enumerate(train_loader))

if __name__ == "__main__":
    main()
