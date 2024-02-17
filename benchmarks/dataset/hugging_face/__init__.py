import logging
from typing import Dict, List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from pfl.data.pytorch import PyTorchDataDataset, PyTorchFederatedDataset

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


class ListDictDataset(torch.utils.data.Dataset):
    """List of dictionary of tensors."""

    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i]


class DictDataset(torch.utils.data.Dataset):
    """ Dictionary of tensors."""

    def __init__(self, data: Dict[str, torch.Tensor]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.data.items()}


class UserDataset(torch.utils.data.Dataset):

    def __init__(self, user_dataset: Dict[str, List[Dict[str, torch.Tensor]]]):
        super().__init__()
        self.user_dataset = user_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, u) -> List[Dict[str, torch.Tensor]]:
        return self.user_dataset[u]


def add_special_tokens(tokenizer: PreTrainedTokenizer):
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"  # noqa: S105
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"  # noqa: S105
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"  # noqa: S105
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"  # noqa: S105
    return tokenizer.add_special_tokens(special_tokens_dict)


def smart_embedding_resize(
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


class HuggingFaceFederatedDataset(PyTorchFederatedDataset):

    def _tensors_to_pfl_dataset(self, tensors):
        assert isinstance(tensors, Dict)
        return PyTorchDataDataset(DictDataset(tensors), **self._dataset_kwargs)
