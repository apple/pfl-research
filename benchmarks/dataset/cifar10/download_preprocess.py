# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Download the CIFAR10 dataset from
https://www.cs.toronto.edu/~kriz/cifar.html
and preprocess into pickles, one for train set and
one for test set.
"""
import argparse
from glob import glob
import os
import pickle
import subprocess
import urllib3

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def dl_preprocess_and_dump(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    raw_output_dir = os.path.join(output_dir, 'raw_data')
    os.makedirs(raw_output_dir, exist_ok=True)

    tar_path = os.path.join(raw_output_dir, "cifar-10-python.tar.gz")
    if not os.path.exists(tar_path):
        print(f'Downloading from {URL}.')
        http = urllib3.PoolManager()
        response = http.request('GET', URL)

        # Save the downloaded file.
        with open(tar_path, 'wb') as f:
            f.write(response.data)
        print(f'Saved raw data to {tar_path}.')
    else:
        print(f'Found {tar_path} on disk, skip download')

    # Extract tar file.
    subprocess.run(f"tar -zxf {tar_path} -C {raw_output_dir}".split(),
                   check=True)

    # Merge all files into a pickle with train data and a pickle with test data.
    def merge_data_and_dump(data_paths, output_file_name):
        print(f'Merging files {data_paths}.')
        images, labels = [], []
        for train_file in data_paths:
            with open(train_file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
                images.append(data[b'data'])
                labels.append(data[b'labels'])
        images = np.concatenate(images).reshape((-1, 32, 32, 3))
        labels = np.concatenate(labels).reshape((-1, 1))
        # This snippet was used to generate cifar10 for ci
        #images = images[:300]
        #labels = labels[:300]
        out_file_path = os.path.join(output_dir, output_file_name)
        with open(out_file_path, 'wb') as f:
            pickle.dump((images, labels), f)
        print(f'Saved processed data to {out_file_path}')

    merge_data_and_dump(glob(raw_output_dir + '/**/data_batch*'),
                        'cifar10_train.p')
    merge_data_and_dump(glob(raw_output_dir + '/**/test_batch*'),
                        'cifar10_test.p')


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--output_dir',
        help=('Output directory for the original files '
              'and the processed pickle files.'),
        default='./data/cifar10')
    arguments = argument_parser.parse_args()

    dl_preprocess_and_dump(arguments.output_dir)
