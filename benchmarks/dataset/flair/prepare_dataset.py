# Copyright © 2023-2024 Apple Inc.
import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, Tuple

import h5py
import numpy as np
import tqdm
from PIL import Image

logger = logging.getLogger(name=__name__)

LABEL_DELIMITER = '|'  # Labels will be joined by delimiter and saved to hdf5
LOG_INTERVAL = 100  # Log the preprocessing progress every interval steps


def load_user_metadata_and_label_counters(
        labels_file: str) -> Tuple[Dict, Counter, Counter]:
    """
    Load labels and metadata keyed by `user_id`, and label counts.

    :param labels_file:
        A .json file with a list of labels and metadata dictionaries. Each
        dictionary has keys: `[image_id,user_id,labels,fine_grained_labels]`.
        * `image_id` is the ID of an image.
        * `user_id` is the ID of the user `image_id` belongs to.
        * `labels` is a list of 17 higher-order class labels.
        * `fine_grained_labels` is a list of 1,628 fine-grained class labels.
    :return:
        Three dictionaries. First dictionary has key being `user_id` and value
        being a list of labels and metadata for each image `user_id` owns.
        Second and third dictionaries are counts for the labels for coarse-grained
        and fine-grained taxonomies.
    """
    user_metadata = defaultdict(list)
    with open(labels_file) as f:
        metadata_list = json.load(f)

    label_counter: Counter = Counter()
    fine_grained_label_counter: Counter = Counter()
    for metadata in metadata_list:
        user_metadata[metadata["user_id"]].append(metadata)
        label_counter.update(metadata["labels"])
        fine_grained_label_counter.update(metadata["fine_grained_labels"])
    return user_metadata, label_counter, fine_grained_label_counter


def preprocess_federated_dataset(image_dir: str, labels_file: str,
                                 output_file: str):
    """
    Process images and labels into a HDF5 federated dataset where data is
    first split by train/test partitions and then split again by user ID.

    :param image_dir:
        Path to directory of images output from the script
        `download_dataset.sh`.
    :param labels_file:
        A .json file with a list of labels and metadata dictionaries. Each
        dictionary has keys: `[image_id,user_id,labels,fine_grained_labels]`.
        * `image_id` is the ID of an image.
        * `user_id` is the ID of the user `image_id` belongs to.
        * `labels` is a list of 17 higher-order class labels.
        * `fine_grained_labels` is a list of ~1,600 fine-grained class labels.
    :param output_file:
        Output path for HDF5 file. Use the postfix `.hdf5`.
    """
    logger.info('Preprocessing federated dataset.')
    (user_metadata, label_counter, fine_grained_label_counter
     ) = load_user_metadata_and_label_counters(labels_file)

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
            # This snippet was used to generate flair_federated_small.h5
            #if i > len(user_metadata)*0.01:
            #    break
            #if len(user_metadata[user_id]) > 20:
            #    # Skip large users
            #    continue
            # This snippet was used to generate flair_federated_ci.h5
            #if i > 12:
            #    break
            #if i > 10:
            #    user_metadata[user_id][0]['partition'] = 'test'

            # Load and concatenate all images of a user.
            image_array, image_id_array = [], []
            labels_row, labels_col = [], []
            fine_grained_labels_row, fine_grained_labels_col = [], []
            # Load and concatenate all images and labels of a user.
            for j, metadata in enumerate(user_metadata[user_id]):
                image_id = metadata["image_id"]
                image = Image.open(os.path.join(image_dir, f"{image_id}.jpg"))
                image_array.append(np.asarray(image))
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


def preprocess_central_dataset(image_dir: str, labels_file: str,
                               output_file: str):
    """
    Process images and labels into a HDF5 (not federated) dataset where
    data is split by train/val/test partitions.

    Same parameters as `preprocess_federated_dataset`.
    """
    logger.info('Preprocessing central dataset.')
    (user_metadata, _, _) = load_user_metadata_and_label_counters(labels_file)
    label_counter: Counter = Counter()
    fine_grained_label_counter: Counter = Counter()
    with h5py.File(output_file, 'w') as h5file:
        # Iterate through users of each partition.
        for i, user_id in enumerate(user_metadata):
            # Load and concatenate all images of a user.
            for metadata in user_metadata[user_id]:
                image_id = metadata["image_id"]
                image = Image.open(os.path.join(image_dir, f"{image_id}.jpg"))
                partition = metadata["partition"]
                h5file.create_dataset(f'/{partition}/{image_id}/image',
                                      data=np.asarray(image))
                # Encode labels as a single string, separated by delimiter |
                h5file[
                    f'/{partition}/{image_id}/labels'] = LABEL_DELIMITER.join(
                        metadata["labels"])
                h5file[f'/{partition}/{image_id}/fine_grained_labels'] = (
                    LABEL_DELIMITER.join(metadata["fine_grained_labels"]))
                h5file[f'/{partition}/{image_id}/user_id'] = user_id
                # Update label counter
                label_counter.update(metadata["labels"])
                fine_grained_label_counter.update(
                    metadata["fine_grained_labels"])

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info(f"Processed {i + 1}/{len(user_metadata)} users")

        # Write metadata
        h5file['/metadata/label_mapping'] = json.dumps(label_counter)
        h5file['/metadata/fine_grained_label_mapping'] = json.dumps(
            fine_grained_label_counter)

    logger.info('Finished preprocessing central dataset successfully!')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    argument_parser = argparse.ArgumentParser(
        description=
        'Preprocess the images and labels of FLAIR dataset into HDF5 files.')
    argument_parser.add_argument(
        '--dataset_dir',
        required=True,
        help='Path to directory of images and label file. '
        'Can be downloaded using download_dataset.py')
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

    image_dir = os.path.join(arguments.dataset_dir, "small_images")
    labels_file = os.path.join(arguments.dataset_dir,
                               "labels_and_metadata.json")
    if arguments.not_group_data_by_user:
        preprocess_central_dataset(image_dir, labels_file,
                                   arguments.output_file)
    else:
        preprocess_federated_dataset(image_dir, labels_file,
                                     arguments.output_file)
