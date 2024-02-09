import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from pfl.data import ArtificialFederatedDataset
from pfl.data.pytorch import PyTorchDataDataset
from pfl.data.sampling import get_data_sampler

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input":
    ("Below is an instruction that describes a task, paired with an input that provides further context. "
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


class AlpacaDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, input_ids: Sequence, labels: Sequence):
        super().__init__()
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


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


def make_iid_federated_dataset(input_ids: Sequence, labels: Sequence,
                               tokenizer: PreTrainedTokenizer,
                               user_dataset_len_sampler: Callable):
    data_collator = AlpacaDataCollator(tokenizer)

    def make_dataset_fn(indices: List):
        user_input_ids, user_labels = [], []
        for i in indices:
            user_input_ids.append(input_ids[i])
            user_labels.append(labels[i])
        return PyTorchDataDataset(raw_data=AlpacaDataset(
            user_input_ids, user_labels),
                                  collate_fn=data_collator)

    data_sampler = get_data_sampler("random", max_bound=len(input_ids))
    return ArtificialFederatedDataset(make_dataset_fn, data_sampler,
                                      user_dataset_len_sampler)


def smart_tokenizer_and_embedding_resize(
    num_new_tokens: int,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    logger.info(f"Resizing model's token embedding to {len(tokenizer)} "
                f"with {num_new_tokens} new tokens.")
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_alpaca_iid_federated_datasets(
    tokenizer: PreTrainedTokenizer,
    user_dataset_len_sampler: Callable,
):
    hf_dataset = load_dataset("tatsu-lab/alpaca")["train"]

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"  # noqa: S105
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"  # noqa: S105
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"  # noqa: S105
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"  # noqa: S105
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    input_ids, labels = preprocess_alpaca(hf_dataset, tokenizer)
    federated_dataset = make_iid_federated_dataset(input_ids, labels,
                                                   tokenizer,
                                                   user_dataset_len_sampler)

    def postprocessing_model_fn(model):
        smart_tokenizer_and_embedding_resize(num_new_tokens, tokenizer, model)

    metadata = {
        'num_new_tokens': num_new_tokens,
        'postprocessing_model_fn': postprocessing_model_fn,
    }
    return federated_dataset, None, None, metadata
