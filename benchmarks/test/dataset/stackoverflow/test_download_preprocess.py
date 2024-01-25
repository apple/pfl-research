# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import json
import os
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from pfl.internal.ops import get_tf_major_version

if get_tf_major_version():
    import tensorflow as tf


def mock_word_counts(vocab_size):
    # vocabulary is alphabet.
    return {chr(97 + i): i for i in range(vocab_size)}


def mock_fetch_client_ids(database_filepath, partition):
    return {"train": [1], "val": [2], "test": [3]}[partition]


def mock_query_client_dataset(database_filepath, user_id, partition):
    sentence_data = {'tokens': tf.constant("a b c")}
    # Number of mock sentences == user_id value
    return tf.data.Dataset.from_generator(
        lambda: iter([sentence_data] * user_id),
        output_types={'tokens': tf.string})


# Only run if TF2 is installed. Using tf.data.
@pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
@patch("dataset.stackoverflow.download_preprocess.load_word_counts",
       mock_word_counts)
@patch("dataset.stackoverflow.download_preprocess.fetch_client_ids",
       mock_fetch_client_ids)
@patch("dataset.stackoverflow.download_preprocess.query_client_dataset",
       mock_query_client_dataset)
@patch("dataset.stackoverflow.download_preprocess.fetch_lzma_file",
       lambda origin, filename: None)
class TestDownloadPreprocess:

    def test_get_vocabulary(self):
        from dataset.stackoverflow.download_preprocess import get_vocabulary

        vocab = get_vocabulary(3)
        assert vocab == {
            'PAD': 0,
            'a': 1,
            'b': 2,
            'c': 3,
            'UNK': 4,
            'BOS': 5,
            'EOS': 6
        }

    def test_dl_preprocess_and_dump_h5(self, tmp_path):
        from dataset.stackoverflow.download_preprocess import dl_preprocess_and_dump_h5
        vocab_size = 3
        max_sequence_length = 2
        num_processes = 2
        dl_preprocess_and_dump_h5(tmp_path, vocab_size, max_sequence_length,
                                  num_processes)

        with h5py.File(os.path.join(tmp_path, 'stackoverflow.hdf5'), 'r') as f:
            # trimmed, max_sequence_length is 3.
            assert json.loads(f['metadata/num_tokens/train'][()]) == {
                str(1): 2
            }
            assert json.loads(f['metadata/num_tokens/val'][()]) == {str(2): 4}
            assert json.loads(f['metadata/num_tokens/test'][()]) == {str(3): 6}
            assert f['metadata/pad_symbol'][()] == 0
            assert f['metadata/unk_symbol'][()] == 4
            assert len(f['metadata/vocabulary'].keys()) == 7
            assert f['metadata/vocabulary_size'][()] == 7
            assert len(f['train'].keys()) == 1
            assert len(f['val'].keys()) == 1
            assert len(f['test'].keys()) == 1
            np.testing.assert_array_equal(f['train/1/inputs'][()],
                                          np.array([[5, 1]]))
            np.testing.assert_array_equal(f['train/1/targets'][()],
                                          np.array([[1, 2]]))
            np.testing.assert_array_equal(f['val/2/inputs'][()],
                                          np.tile([5, 1], (2, 1)))
            np.testing.assert_array_equal(f['val/2/targets'][()],
                                          np.tile([1, 2], (2, 1)))
            np.testing.assert_array_equal(f['test/3/inputs'][()],
                                          np.tile([5, 1], (3, 1)))
            np.testing.assert_array_equal(f['test/3/targets'][()],
                                          np.tile([1, 2], (3, 1)))
