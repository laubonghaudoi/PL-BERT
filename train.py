#!/usr/bin/env python3
import os
import shutil
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from transformers import AdamW, AlbertConfig, AlbertModel, AutoTokenizer
from model import MultiTaskModel
from dataloader import build_dataloader
from utils import length_to_mask, scan_checkpoint

from datasets import load_from_disk

from torch.utils.tensorboard import SummaryWriter

import yaml
import pickle

# Load configuration
config_path = "Configs/config.yml"  # you can change it to anything else
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load token maps (generated during preprocessing)
# token_maps.pkl is produced by preprocess.py and maps tokens to lower-cased token ids.
with open(config['dataset_params']['token_maps'], 'rb') as handle:
    token_maps = pickle.load(handle)

# Updated tokenizer loading using AutoTokenizer
os.environ["TRUST_REMOTE_CODE"] = "True"
tokenizer = AutoTokenizer.from_pretrained(config['dataset_params']['tokenizer'], use_fast=False)

# Training hyperparameters
criterion = nn.CrossEntropyLoss()  # F0 loss (regression)
num_steps = config['num_steps']
log_interval = config['log_interval']
save_interval = config['save_interval']

def train():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    curr_steps = 0

    # Load preprocessed dataset
    dataset = load_from_disk(config["data_folder"])

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    batch_size = config["batch_size"]
    train_loader = build_dataloader(dataset,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    dataset_config=config['dataset_params'])

    # Define model using Albert as backbone
    albert_config = AlbertConfig(**config['model_params'])
    base_model = AlbertModel(albert_config)
    # The number of vocabulary tokens used in the model is computed from token_maps.pkl.
    num_vocab = 1 + max([m['token'] for m in token_maps.values()])
    bert = MultiTaskModel(base_model,
                          num_vocab=num_vocab,
                          num_tokens=config['model_params']['vocab_size'],
                          hidden_size=config['model_params']['hidden_size'])

    # Load latest checkpoint if available
    load_checkpoint = True
    try:
        ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        latest_iter = sorted([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files
                                if osp.isfile(osp.join(log_dir, f))])[-1]
    except Exception:
        latest_iter = 0
        load_checkpoint = False

    optimizer = AdamW(bert.parameters(), lr=1e-4)
    accelerator = Accelerator(mixed_precision=config['mixed_precision'],
                              split_batches=True,
                              kwargs_handlers=[ddp_kwargs])

    if load_checkpoint:
        checkpoint_path = osp.join(log_dir, "step_" + str(latest_iter) + ".t7")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['net']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove the `module.` prefix, if present
            new_state_dict[name] = v
        bert.load_state_dict(new_state_dict, strict=False)
        accelerator.print('Checkpoint loaded.')
        optimizer.load_state_dict(checkpoint['optimizer'])

    bert, optimizer, train_loader = accelerator.prepare(bert, optimizer, train_loader)
    accelerator.print('Start training...')

    running_loss = 0
    current_iter = latest_iter

    for _, batch in enumerate(train_loader):
        curr_steps += 1
        words, labels, phonemes, input_lengths, masked_indices = batch
        text_mask = length_to_mask(torch.Tensor(input_lengths)).to(phonemes.device)
        tokens_pred, words_pred = bert(phonemes, attention_mask=(~text_mask).int())

        loss_vocab = 0
        for _s2s_pred, _text_input, _text_length, _masked_indices in zip(words_pred, words, input_lengths, masked_indices):
            loss_vocab += criterion(_s2s_pred[:_text_length], _text_input[:_text_length])
        loss_vocab /= words.size(0)

        loss_token = 0
        sizes = 1
        for _s2s_pred, _text_input, _text_length, _masked_indices in zip(tokens_pred, labels, input_lengths, masked_indices):
            if len(_masked_indices) > 0:
                _text_input_masked = _text_input[:_text_length][_masked_indices]
                loss_tmp = criterion(_s2s_pred[:_text_length][_masked_indices], _text_input_masked)
                loss_token += loss_tmp
                sizes += 1
        loss_token /= sizes

        loss = loss_vocab + loss_token

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        running_loss += loss.item()
        current_iter += 1

        if (current_iter + 1) % log_interval == 0:
            accelerator.print('Step [%d/%d], Loss: %.5f, Vocab Loss: %.5f, Token Loss: %.5f'
                              % (current_iter + 1, num_steps, running_loss / log_interval, loss_vocab, loss_token))
            running_loss = 0

        if (current_iter + 1) % save_interval == 0:
            accelerator.print('Saving checkpoint...')
            state = {
                'net': bert.state_dict(),
                'step': current_iter,
                'optimizer': optimizer.state_dict(),
            }
            save_path = osp.join(log_dir, 'step_' + str(current_iter + 1) + '.t7')
            accelerator.save(state, save_path)

        if curr_steps > num_steps:
            break

if __name__ == "__main__":
    train()
