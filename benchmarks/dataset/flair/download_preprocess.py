# Copyright © 2024 Apple Inc.
import argparse
import itertools
import json
import logging
import sys
from collections import Counter, defaultdict
from typing import Dict, Tuple

import h5py
import numpy as np
import tqdm
from datasets import load_dataset

logger = logging.getLogger(name=__name__)

LABEL_DELIMITER = '|'  # Labels will be joined by delimiter and saved to hdf5
LOG_INTERVAL = 100  # Log the preprocessing progress every interval steps

def load_image_from_huggingface(dataset, image_id):
    """
    Load an image from the HuggingFace dataset by image_id.

    :param dataset:
        The loaded HuggingFace dataset.
    :param image_id:
        The image_id of the image to be loaded.
    :return:
        The loaded image as a PIL Image object.
    """
    # Assuming image_id is unique and using filter to get the specific image
    image_data = dataset.filter(lambda x: x['image_id'] == image_id)
    image_entry = next(iter(image_data))  # Get the first (and only) entry
    return image_entry['image']


def preprocess_federated_dataset(output_file: str):
    """
    Process images and labels into a HDF5 federated dataset where data is
    first split by train/test partitions and then split again by user ID.

    :param dataset:
        The loaded HuggingFace dataset.
    :param output_file:
        Output path for HDF5 file. Use the postfix `.hdf5`.
    """

    # Load dataset from HuggingFace
    # This is a Dict[str, Dataset] where key is the split.
    dataset_splits = load_dataset('apple/flair')
    logger.info(f'Preprocessing federated dataset, sample record: {next(iter(dataset_splits["train"]))}')

    user_metadata = defaultdict(list)
    label_counter = Counter()
    fine_grained_label_counter = Counter()

    
    for i, entry in enumerate(tqdm.tqdm(
        itertools.chain(*dataset_splits.values()),
        total=sum(map(len, dataset_splits.values())),
        desc='preparing users')):
        user_metadata[entry['user_id']].append(entry)
        label_counter.update(entry["labels"])
        fine_grained_label_counter.update(entry["fine_grained_labels"])
        if i == 100:
            break

    label_to_index = {
        label: index
        for index, label in enumerate(sorted(label_counter.keys()))
    }
    fine_grained_label_to_index = {
        fine_grained_label: index
        for index, fine_grained_label in enumerate(
            sorted(fine_grained_label_counter.keys()))
    }

    label_counter = Counter()
    fine_grained_label_counter = Counter()
    with h5py.File(output_file, 'w') as h5file:
        # Iterate through users of each partition.
        for i, user_id in tqdm.tqdm(enumerate(user_metadata),
                                    total=len(user_metadata)):
            # Load and concatenate all images and labels of a user.
            image_array, image_id_array = [], []
            labels_row, labels_col = [], []
            fine_grained_labels_row, fine_grained_labels_col = [], []
            for j, metadata in enumerate(user_metadata[user_id]):
                image_id = metadata["image_id"]
                image_array.append(np.asarray(metadata["image"]))
                image_id_array.append(image_id)
                # Encode labels as row indices and column indices
                labels_row.extend([j] * len(metadata["labels"]))
                labels_col.extend(
                    [label_to_index[l] for l in metadata["labels"]])
                fine_grained_labels_row.extend(
                    [j] * len(metadata["fine_grained_labels"]))
                fine_grained_labels_col.extend([
                    fine_grained_label_to_index[l]
                    for l in metadata["fine_grained_labels"]
                ])
                # Update label counter
                label_counter.update(metadata["labels"])
                fine_grained_label_counter.update(
                    metadata["fine_grained_labels"])

            partition = user_metadata[user_id][0]["partition"]
            # Multiple variable-length labels. Needs to be stored as a string.
            h5file[f'/{partition}/{user_id}/labels_row'] = np.asarray(
                labels_row, dtype=np.uint16)
            h5file[f'/{partition}/{user_id}/labels_col'] = np.asarray(
                labels_col, dtype=np.uint8)
            h5file[
                f'/{partition}/{user_id}/fine_grained_labels_row'] = np.asarray(
                    fine_grained_labels_row, dtype=np.uint16)
            h5file[
                f'/{partition}/{user_id}/fine_grained_labels_col'] = np.asarray(
                    fine_grained_labels_col, dtype=np.uint16)
            h5file[f'/{partition}/{user_id}/image_ids'] = np.asarray(
                image_id_array, dtype='S')
            # Tensor with dimensions [num_images,width,height,channels]
            h5file.create_dataset(f'/{partition}/{user_id}/images',
                                  data=np.stack(image_array))

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(user_metadata)} users")

        # Write metadata
        h5file['/metadata/label_mapping'] = json.dumps(label_to_index)
        h5file['/metadata/fine_grained_label_mapping'] = json.dumps(
            fine_grained_label_to_index)

    logger.info('Finished preprocess federated dataset successfully!')


def preprocess_central_dataset(output_file: str):
    """
    Process images and labels into a HDF5 (not federated) dataset where
    data is split by train/val/test partitions.

    :param dataset:
        The loaded HuggingFace dataset.
    :param output_file:
        Output path for HDF5 file. Use the postfix `.hdf5`.
    """
    logger.info('Preprocessing central dataset.')
    
    label_counter = Counter()
    fine_grained_label_counter = Counter()
    with h5py.File(output_file, 'w') as h5file:
        # Iterate through dataset.
        for i, entry in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            image_id = entry["image_id"]
            image = np.asarray(entry["image"])  # Directly use the image array from the dataset
            partition = entry["partition"]
            h5file.create_dataset(f'/{partition}/{image_id}/image',
                                  data=image)
            # Encode labels as a single string, separated by delimiter |
            h5file[f'/{partition}/{image_id}/labels'] = LABEL_DELIMITER.join(
                entry["labels"])
            h5file[f'/{partition}/{image_id}/fine_grained_labels'] = (
                LABEL_DELIMITER.join(entry["fine_grained_labels"]))
            h5file[f'/{partition}/{image_id}/user_id'] = entry["user_id"]
            # Update label counter
            label_counter.update(entry["labels"])
            fine_grained_label_counter.update(entry["fine_grained_labels"])

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} entries")

        # Write metadata
        h5file['/metadata/label_mapping'] = json.dumps(dict(label_counter))
        h5file['/metadata/fine_grained_label_mapping'] = json.dumps(
            dict(fine_grained_label_counter))

    logger.info('Finished preprocessing central dataset successfully!')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    argument_parser = argparse.ArgumentParser(
        description=
        'Download and preprocess the images and labels of FLAIR dataset into HDF5 files.')
    argument_parser.add_argument(
        '--output_file',
        required=True,
        help='Path to output HDF5 file that will be constructed by this script'
    )
    argument_parser.add_argument('--not_group_data_by_user',
                                 action='store_true',
                                 default=False,
                                 help='If true, do not group data by user IDs.'
                                 'If false, group data by user IDs to '
                                 'make suitable for federated learning.')
    arguments = argument_parser.parse_args()

    if arguments.not_group_data_by_user:
        preprocess_central_dataset(arguments.output_file)
    else:
        preprocess_federated_dataset(arguments.output_file)

