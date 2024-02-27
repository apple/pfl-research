# Copyright © 2023-2024 Apple Inc.

import logging
import math
from collections import defaultdict
from typing import Dict, List

import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from . import IGNORE_INDEX
from .oasst import encode, make_central_dataset, make_federated_dataset, pad

logger = logging.getLogger(__name__)


def preprocess_aya_for_causal_lm(
        dataset: HuggingFaceDataset, tokenizer: PreTrainedTokenizer,
        max_user_instructions: int) -> Dict[str, List[Dict]]:
    """
    Preprocess the Aya dataset by tokenizing the instructions and outputs.
    """
    max_length = tokenizer.model_max_length // 2 - 1  # minus 1 to include EOS
    user_dataset = defaultdict(list)
    # Reusing the same preprocessing as OpenAssistant for now
    for row in dataset:
        instruction, output = row['inputs'], row['targets']
        instruction_ids = encode(instruction, tokenizer, max_length)
        output_ids = encode(output, tokenizer, max_length)
        input_ids = torch.cat([instruction_ids, output_ids])
        labels = input_ids.clone()
        instruction_mask = torch.cat(
            [torch.ones_like(instruction_ids),
             torch.zeros_like(output_ids)]).bool()
        labels.masked_fill_(instruction_mask, IGNORE_INDEX)
        input_ids, attention_masks = pad(input_ids, tokenizer.pad_token_id,
                                         tokenizer.model_max_length)
        user_dataset[row['user_id']].append({
            "input_ids":
            input_ids,
            "attention_masks":
            attention_masks,
            "labels":
            pad(labels, IGNORE_INDEX, tokenizer.model_max_length)[0],
        })

    for user_id in list(user_dataset.keys()):
        local_dataset = user_dataset[user_id]
        if len(local_dataset) > max_user_instructions:
            # Split the power user's local dataset into multiple subsets
            user_dataset[user_id] = local_dataset[:max_user_instructions]
            extra_subsets = int(
                math.ceil(len(local_dataset) / max_user_instructions)) - 1
            for i in range(1, extra_subsets + 1):
                subset = local_dataset[i * max_user_instructions:(i + 1) *
                                       max_user_instructions]
                user_dataset[f"{user_id}-{i}"] = subset

    return user_dataset


def make_aya_datasets(tokenizer: PreTrainedTokenizer,
                      max_user_instructions: int,
                      dataloader_kwargs: Dict,
                      train_split_ratio: float = 0.95):
    """
    Create a train and test ``FederatedDataset`` as well as a
    central dataset for Aya dataset.
    """
    hf_dataset = load_dataset("CohereForAI/aya_dataset", split="train+test")
    user_dataset = preprocess_aya_for_causal_lm(hf_dataset, tokenizer,
                                                max_user_instructions)
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
