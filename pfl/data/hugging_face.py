from typing import Callable

from .dataset import Dataset
from .federated_dataset import ArtificialFederatedDataset


class HuggingFaceDataset(Dataset):
    pass


class HuggingFaceArtificialFederatedDataset(ArtificialFederatedDataset):

    @classmethod
    def from_slices(cls,
                    data,
                    data_sampler,
                    sample_dataset_len,
                    create_dataset_fn: Callable = lambda data: Dataset(data)):
        raise NotImplementedError()

    @classmethod
    def from_hf_hub(cls, hf_hub_path: str, data_sampler, sample_dataset_len,
                    create_dataset_fn, **kwargs):
        from datasets import load_dataset

        hf_dataset = load_dataset(hf_hub_path, **kwargs)
        print(hf_dataset)


if __name__ == '__main__':
    HuggingFaceArtificialFederatedDataset.from_hf_hub("tatsu-lab/alpaca", None,
                                                      None, None)
