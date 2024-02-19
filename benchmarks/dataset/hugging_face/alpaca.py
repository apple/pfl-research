"""
Some of the code is adapted from: https://github.com/tatsu-lab/stanford_alpaca
"""

import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from pfl.data.pytorch import PyTorchDataDataset
from pfl.data.sampling import get_user_sampler

from . import (
    IGNORE_INDEX,
    GetItemDataset,
    HuggingFaceFederatedDataset,
)

logger = logging.getLogger(__name__)

PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input "
     "that provides further context. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
     ),
    "prompt_no_input":
    ("Below is an instruction that describes a task. "
     "Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Response:"),
}


def _tokenize_alpaca(strings: Sequence[str],
                     tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def preprocess_alpaca(hf_dataset, tokenizer: PreTrainedTokenizer) -> Tuple:
    """Preprocess the data by tokenizing."""
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT[
        "prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else
        prompt_no_input.format_map(example) for example in hf_dataset
    ]
    targets = [
        f"{example['output']}{tokenizer.eos_token}" for example in hf_dataset
    ]

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (_tokenize_alpaca(
        strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels


@dataclass
class AlpacaDataCollator:
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def iid_user_partition(
        input_ids: Sequence, labels: Sequence,
        user_dataset_len_sampler: Callable) -> Dict[str, List[Dict]]:
    start_ix = 0
    users_to_data: Dict = {}
    while True:
        dataset_len = user_dataset_len_sampler()
        dataset_len = min(dataset_len, len(input_ids) - start_ix)
        user_data = [{
            "input_ids": input_ids[i],
            "labels": labels[i]
        } for i in range(start_ix, start_ix + dataset_len)]
        users_to_data[str(len(users_to_data))] = user_data
        start_ix += dataset_len
        if start_ix >= len(input_ids):
            break
    return users_to_data


def make_iid_federated_dataset(user_dataset: Dict[str, List[Dict]],
                               tokenizer: PreTrainedTokenizer,
                               dataloader_kwargs: Dict):
    user_sampler = get_user_sampler('random', list(user_dataset.keys()))
    user_id_to_weight = {k: len(v) for k, v in user_dataset.items()}
    return HuggingFaceFederatedDataset(
        GetItemDataset(user_dataset),
        user_sampler,
        user_id_to_weight=user_id_to_weight,
        batch_size=None,
        collate_fn=AlpacaDataCollator(tokenizer),
        **dataloader_kwargs)


def make_central_dataset(user_dataset: Dict[str, List[Dict]],
                         tokenizer: PreTrainedTokenizer):
    list_dataset = []
    for u in user_dataset:
        list_dataset += user_dataset[u]
    return PyTorchDataDataset(raw_data=GetItemDataset(list_dataset),
                              collate_fn=AlpacaDataCollator(tokenizer))


def make_alpaca_iid_datasets(tokenizer: PreTrainedTokenizer,
                             user_dataset_len_sampler: Callable,
                             dataloader_kwargs: Dict,
                             train_split_ratio: float = 0.9):
    hf_dataset = load_dataset("tatsu-lab/alpaca")["train"]

    input_ids, labels = preprocess_alpaca(hf_dataset, tokenizer)
    user_dataset = iid_user_partition(input_ids, labels,
                                      user_dataset_len_sampler)
    users = list(user_dataset.keys())
    num_train_users = int(train_split_ratio * len(users))
    train_user_dataset = {u: user_dataset[u] for u in users[:num_train_users]}
    val_user_dataset = {u: user_dataset[u] for u in users[num_train_users:]}

    train_federated_dataset = make_iid_federated_dataset(
        train_user_dataset, tokenizer, dataloader_kwargs)
    val_federated_dataset = make_iid_federated_dataset(val_user_dataset,
                                                       tokenizer,
                                                       dataloader_kwargs)
    central_dataset = make_central_dataset(val_user_dataset, tokenizer)
    logger.info(f"# of train users = {len(train_user_dataset)}, "
                f"# of val users = {len(val_user_dataset)}")

    return train_federated_dataset, val_federated_dataset, central_dataset, {}
