"""
Some of the code is adapted from: https://github.com/h2oai/h2o-llmstudio/
"""

import logging
from collections import defaultdict
from typing import Dict, List

import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer, default_data_collator

from pfl.data.pytorch import PyTorchDataDataset
from pfl.data.sampling import get_user_sampler

from . import (
    IGNORE_INDEX,
    GetItemDataset,
    HuggingFaceFederatedDataset,
)

logger = logging.getLogger(__name__)


def append_eos(encoding: torch.Tensor, eos_token_id: int):
    return torch.cat([encoding, torch.tensor([eos_token_id])], dim=0)


def encode(text, tokenizer, max_length):
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return append_eos(encodings["input_ids"][0][:max_length],
                      tokenizer.eos_token_id)


def pad(token_ids, pad_token_id, max_length):
    attention_mask = torch.zeros(max_length)
    attention_mask[-len(token_ids):] = torch.ones_like(token_ids)
    if len(token_ids) < max_length:
        padded_token_ids = torch.full((max_length, ), pad_token_id)
        padded_token_ids[-len(token_ids):] = token_ids
        return padded_token_ids, attention_mask
    return token_ids, attention_mask


def preprocess_oasst_for_causal_lm(
        dataset: HuggingFaceDataset,
        tokenizer: PreTrainedTokenizer) -> Dict[str, List[Dict]]:
    df = dataset.to_pandas()
    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    user_ids = df_assistant["user_id"].values
    outputs = df_assistant["text"].values
    instructions = []
    for _, row in df_assistant.iterrows():
        instructions.append(df_prompter.loc[row.parent_id].text)

    max_length = tokenizer.model_max_length // 2 - 1  # minus 1 to include EOS

    user_dataset = defaultdict(list)
    for user_id, instruction, output in zip(user_ids, instructions, outputs):
        instruction_ids = encode(instruction, tokenizer, max_length)
        output_ids = encode(output, tokenizer, max_length)
        input_ids = torch.cat([instruction_ids, output_ids])
        labels = input_ids.clone()
        instruction_mask = torch.cat(
            [torch.ones_like(instruction_ids),
             torch.zeros_like(output_ids)])
        labels.masked_fill_(instruction_mask, IGNORE_INDEX)
        input_ids, attention_masks = pad(input_ids, tokenizer.pad_token_id,
                                         tokenizer.model_max_length)
        user_dataset[user_id].append({
            "input_ids":
            input_ids,
            "attention_masks":
            attention_masks,
            "labels":
            pad(labels, IGNORE_INDEX, tokenizer.model_max_length)[0],
        })
    return user_dataset


def make_federated_dataset(user_dataset: Dict[str, List[Dict]],
                           dataloader_kwargs: Dict):
    user_sampler = get_user_sampler('random', list(user_dataset.keys()))
    user_id_to_weight = {k: len(v) for k, v in user_dataset.items()}
    return HuggingFaceFederatedDataset(GetItemDataset(user_dataset),
                                       user_sampler,
                                       user_id_to_weight=user_id_to_weight,
                                       batch_size=None,
                                       collate_fn=default_data_collator,
                                       **dataloader_kwargs)


def make_central_dataset(user_dataset: Dict[str, List[Dict]]):
    list_dataset = []
    for u in user_dataset:
        list_dataset += user_dataset[u]
    return PyTorchDataDataset(raw_data=GetItemDataset(list_dataset),
                              collate_fn=default_data_collator)


def make_oasst_datasets(tokenizer: PreTrainedTokenizer,
                        dataloader_kwargs: Dict,
                        train_split_ratio: float = 0.9):
    hf_dataset = load_dataset("OpenAssistant/oasst2", split="train+validation")
    user_dataset = preprocess_oasst_for_causal_lm(hf_dataset, tokenizer)
    users = list(user_dataset.keys())
    num_train_users = int(train_split_ratio * len(users))
    train_user_dataset = {u: user_dataset[u] for u in users[:num_train_users]}
    val_user_dataset = {u: user_dataset[u] for u in users[num_train_users:]}

    train_federated_dataset = make_federated_dataset(train_user_dataset,
                                                     dataloader_kwargs)
    val_federated_dataset = make_federated_dataset(val_user_dataset,
                                                   dataloader_kwargs)
    central_dataset = make_central_dataset(val_user_dataset)
    logger.info(f"# of train users = {len(train_user_dataset)}, "
                f"# of val users = {len(val_user_dataset)}")

    return train_federated_dataset, val_federated_dataset, central_dataset, {}
