# Copyright © 2023-2024 Apple Inc.
import argparse
from typing import Any, Callable, Dict, Tuple

import numpy as np
from utils.argument_parsing import store_bool

from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import FederatedDatasetBase


def add_pytorch_dataloader_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--dataloader_num_workers',
                        type=int,
                        default=0,
                        help='Number of subprocesses to use for data loading.')
    parser.add_argument(
        '--dataloader_pin_memory',
        action=store_bool,
        default=False,
        help='Whether dataloader copy Tensors into device/CUDA '
        'pinned memory before returning them.')
    parser.add_argument('--dataloader_prefetch_factor',
                        type=int,
                        default=None,
                        help='Number of batches loaded in advance by '
                        'each worker.')
    parser.add_argument('--multiprocessing_sharing_strategy',
                        type=str,
                        default=None,
                        choices=['file_descriptor', 'file_system'],
                        help='PyTorch multiprocessing sharing_strategy.')
    return parser


def parse_pytorch_dataloader_kwargs(args: argparse.Namespace) -> Dict:
    if (args.dataloader_num_workers > 0
            and args.multiprocessing_sharing_strategy is not None):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy(
            args.multiprocessing_sharing_strategy)

    return {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "prefetch_factor": args.dataloader_prefetch_factor,
    }


def add_dataset_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add the argument ``dataset`` to parser and add dataset-specific
    arguments depending on the dataset specified in ``dataset``.
    """

    parser.add_argument('--dataset',
                        choices=[
                            'cifar10',
                            'cifar10_iid',
                            'femnist',
                            'femnist_digits',
                            'reddit',
                            'flair',
                            'flair_iid',
                            'flair_pytorch',
                            'stackoverflow',
                            'alpaca',
                            'aya',
                            'oasst',
                        ],
                        default='cifar10',
                        help='Which dataset to train on')

    parser.add_argument('--minimum_num_datapoints_per_user',
                        type=int,
                        default=1,
                        help='The minimum number of samples that allows a'
                        'simulated user to participate.')

    # Get the value of `dataset` argument and dynamically add
    # arguments depending on which dataset is chosen.
    known_args, _ = parser.parse_known_args()

    if known_args.dataset == 'femnist':
        pass
    elif known_args.dataset == 'femnist_digits':
        pass
    elif known_args.dataset in ['cifar10', 'cifar10_iid']:
        parser = add_artificial_fed_dataset_arguments(parser)
        parser.add_argument(
            "--partition_alpha",
            type=float,
            default=0.1,
            help='Alpha for Dir(alpha) non-iid class partitioning, '
            'not used for cifar10_iid')
    elif known_args.dataset in {'reddit', 'stackoverflow'}:
        parser.add_argument("--data_fraction",
                            type=float,
                            default=1.0,
                            help='Fraction of total dataset to use.')

        parser.add_argument(
            "--central_data_fraction",
            type=float,
            default=1.0,
            help='Fraction of dataset to use in central evaluation.')

        parser.add_argument(
            "--max_user_tokens",
            type=int,
            default=1600,
            help='Maximum number of tokens to train on per user in FL.')

        parser.add_argument(
            "--max_user_sentences",
            type=int,
            default=1600,
            help='Maximum number of sentences to train on per user in FL.')

    elif known_args.dataset in ['flair', 'flair_iid', 'flair_pytorch']:
        parser = add_artificial_fed_dataset_arguments(parser)

        parser.add_argument('--use_fine_grained_labels',
                            action=store_bool,
                            default=False,
                            help='Whether to use fine-grained label taxonomy.')

        parser.add_argument('--max_num_user_images',
                            type=int,
                            default=100,
                            help='Maximum number of images per user')

        parser.add_argument(
            '--scheduling_base_weight_multiplier',
            type=float,
            default=1.0,
            help=('Figure 3b in pfl-research paper '
                  'https://arxiv.org/abs/2404.06430 show adding a'
                  'base value for each user\'s weight for scheduling '
                  'in distributed simulations speeds up training. '
                  'This parameter adds a '
                  'multiplicative factor of the median user weight '
                  'as base value. 0.0 means no base value added and '
                  '~1.0 is the optimal value for the FLAIR benchmark '
                  'according to Figure 3b (can be different for '
                  'other setups).'))
    elif known_args.dataset == 'alpaca':
        parser = add_artificial_fed_dataset_arguments(parser)

    elif known_args.dataset == 'aya':
        parser.add_argument('--max_user_instructions',
                            type=int,
                            default=64,
                            help='Maximum number of instructions per user.')

    if known_args.dataset in ['alpaca', 'aya', 'oasst', 'flair_pytorch']:
        parser = add_pytorch_dataloader_arguments(parser)

    return parser


def add_artificial_fed_dataset_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add arguments to parser that only have an effect when using
    ``ArtificialFederatedDataset``, e.g. for CIFAR10.
    """

    parser.add_argument('--datapoints_per_user_distribution',
                        choices=['constant', 'poisson'],
                        default='constant',
                        help='Distribution of the number of samples that '
                        'simulated users have to train on.')

    parser.add_argument('--mean_datapoints_per_user',
                        type=float,
                        default=50,
                        help='Mean parameter for the number of samples that '
                        'the simulated users have to train on.')

    return parser


def parse_draw_num_datapoints_per_user(
        datapoints_per_user_distribution: str,
        mean_datapoints_per_user: float,
        minimum_num_datapoints_per_user: int = 1) -> Callable[[], int]:
    """
    Get a user dataset length sampler.

    :param datapoints_per_user_distribution:
        'constant' or 'poisson'.
    :param mean_datapoints_per_user:
        If 'constant' distribution, this is the value to always return.
        If 'poisson' distribution, this is the mean of the poisson
        distribution.
    :param minimum_num_datapoints_per_user:
        Only accept return values that are at least as large as this argument.
        If rejected, value is resampled until above the threshold, which will
        result in a truncated distribution.
    :return:
        A callable that samples lengths for artificial user datasets
        yet to be created.
    """
    if datapoints_per_user_distribution == 'constant':
        assert minimum_num_datapoints_per_user < mean_datapoints_per_user
        draw_num_datapoints_per_user = lambda: int(mean_datapoints_per_user)
    else:
        assert datapoints_per_user_distribution == 'poisson'

        def draw_truncated_poisson():
            while True:
                num_datapoints = np.random.poisson(mean_datapoints_per_user)
                # Try again if less than minimum specified.
                if num_datapoints >= minimum_num_datapoints_per_user:
                    return num_datapoints

        draw_num_datapoints_per_user = draw_truncated_poisson
    return draw_num_datapoints_per_user


def get_datasets(
    args: argparse.Namespace
) -> Tuple[FederatedDatasetBase, FederatedDatasetBase, Dataset, Dict[str,
                                                                     Any]]:
    """
    Create a federated dataset for training, a federated dataset for evalution
    and a central dataset for central evaluation.

    :param args:
        ``args.dataset`` specifies which dataset to load. Should be one of
        ``{cifar10,femnist,femnist_digits}``.
        ``args`` should also have any dataset-specific arguments added by
        ``add_dataset_arguments`` for the particular ``args.dataset`` chosen.
    :return:
        A tuple ``(fed_train, fed_eval, eval, metadata)``, where ``fed_train``
        is a federated dataset to be used for training, ``fed_eval`` is a
        federated dataset from a population separate for ``fed_train``,
        ``eval`` is a dataset for central evaluation, and ``metadata`` is
        a dictionary of metadata for the particular dataset.
    """
    # create federated training and val datasets from central training and val
    # data
    numpy_to_tensor = getattr(args, "numpy_to_tensor", lambda x: x)
    datasets: Tuple[FederatedDatasetBase, FederatedDatasetBase, Dataset,
                    Dict[str, Any]]
    if args.dataset.startswith('cifar10'):

        from . import cifar10
        user_dataset_len_sampler = parse_draw_num_datapoints_per_user(
            args.datapoints_per_user_distribution,
            args.mean_datapoints_per_user,
            args.minimum_num_datapoints_per_user)

        if args.dataset == 'cifar10':
            datasets = cifar10.make_cifar10_datasets(args.data_path,
                                                     user_dataset_len_sampler,
                                                     numpy_to_tensor,
                                                     args.partition_alpha)
        elif args.dataset == 'cifar10_iid':
            datasets = cifar10.make_cifar10_iid_datasets(
                args.data_path, user_dataset_len_sampler, numpy_to_tensor)
    elif args.dataset.startswith('femnist'):
        from .femnist import make_femnist_datasets
        datasets = make_femnist_datasets(
            args.data_path,
            digits_only=args.dataset == 'femnist_digits',
            numpy_to_tensor=numpy_to_tensor)
    elif args.dataset == 'reddit':
        from .reddit import make_reddit_datasets
        datasets = make_reddit_datasets(
            data_path=args.data_path,
            include_mask=args.include_mask,
            local_batch_size=args.local_batch_size,
            max_user_tokens=args.max_user_tokens,
            data_fraction=args.data_fraction,
            central_data_fraction=args.central_data_fraction,
            minimum_num_datapoints_per_user=args.
            minimum_num_datapoints_per_user)
    elif args.dataset == 'stackoverflow':
        from .stackoverflow.numpy import make_stackoverflow_datasets
        datasets = make_stackoverflow_datasets(
            data_path=args.data_path,
            max_user_sentences=args.max_user_sentences,
            data_fraction=args.data_fraction,
            central_data_fraction=args.central_data_fraction)
    elif args.dataset == 'flair_iid':
        from .flair import make_flair_iid_datasets
        user_dataset_len_sampler = parse_draw_num_datapoints_per_user(
            args.datapoints_per_user_distribution,
            args.mean_datapoints_per_user,
            args.minimum_num_datapoints_per_user)
        datasets = make_flair_iid_datasets(
            data_path=args.data_path,
            use_fine_grained_labels=args.use_fine_grained_labels,
            user_dataset_len_sampler=user_dataset_len_sampler,
            numpy_to_tensor=numpy_to_tensor)
    elif args.dataset == 'flair':
        from .flair import make_flair_datasets
        datasets = make_flair_datasets(
            data_path=args.data_path,
            use_fine_grained_labels=args.use_fine_grained_labels,
            max_num_user_images=args.max_num_user_images,
            scheduling_base_weight_multiplier=args.
            scheduling_base_weight_multiplier,
            numpy_to_tensor=numpy_to_tensor)
    elif args.dataset == 'flair_pytorch':
        from .flair import make_flair_pytorch_datasets
        datasets = make_flair_pytorch_datasets(
            data_path=args.data_path,
            use_fine_grained_labels=args.use_fine_grained_labels,
            max_num_user_images=args.max_num_user_images,
            dataloader_kwargs=parse_pytorch_dataloader_kwargs(args))
    elif args.dataset == 'alpaca':
        from .hugging_face.alpaca import make_alpaca_iid_datasets
        assert hasattr(args, 'tokenizer'), (
            "Hugging Face tokenizer is required to parse Alpaca dataset")
        user_dataset_len_sampler = parse_draw_num_datapoints_per_user(
            args.datapoints_per_user_distribution,
            args.mean_datapoints_per_user,
            args.minimum_num_datapoints_per_user)
        datasets = make_alpaca_iid_datasets(
            args.tokenizer, user_dataset_len_sampler,
            parse_pytorch_dataloader_kwargs(args))
    elif args.dataset == 'aya':
        from .hugging_face.aya import make_aya_datasets
        assert hasattr(args, 'tokenizer'), (
            "Hugging Face tokenizer is required to parse Aya dataset")
        datasets = make_aya_datasets(args.tokenizer,
                                     args.max_user_instructions,
                                     parse_pytorch_dataloader_kwargs(args))
    elif args.dataset == 'oasst':
        from .hugging_face.oasst import make_oasst_datasets
        assert hasattr(args, 'tokenizer'), (
            "Hugging Face tokenizer is required to parse OpenAssistant dataset"
        )
        datasets = make_oasst_datasets(args.tokenizer,
                                       parse_pytorch_dataloader_kwargs(args))
    else:
        raise ValueError(f'{args.dataset} is not supported')
    return datasets
