# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
# Some of the code in this file is adapted from:
#
# tensorflow/federated:
# Copyright 2023, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
"""
Download the StackOverflow dataset
(https://www.kaggle.com/datasets/stackoverflow/stackoverflow)
using code adapted from TensoFlow Federated
(https://www.tensorflow.org/federated/api_docs/python/tff/
simulation/datasets/stackoverflow/load_data)
such that experiment setups match that of other frameworks.
Preprocess and save dataset as HDF5 on disk in ML-ready format.
This script is a one-time procedure.
"""
from collections import defaultdict, OrderedDict
from typing import Dict, Iterator, Optional

import argparse
import json
import os
import sqlite3
import lzma
import urllib.request
import urllib.parse

from tqdm import tqdm

import multiprocess as mp
import h5py
import numpy as np
import tensorflow as tf


PAD = 'PAD'
UNK = 'UNK'
BOS = 'BOS'
EOS = 'EOS'


def load_word_counts(cache_dir=None, vocab_size: Optional[int] = None):
    """Loads the word counts for the Stack Overflow dataset.

    :param: cache_dir:
        (Optional) directory to cache the downloaded file. If `None`,
        caches in Keras' default cache directory.
    :param: vocab_size:
        (Optional) when specified, only load the first `vocab_size`
        unique words in the vocab file (i.e. the most frequent `vocab_size`
        words).

    :returns:
      A collections.OrderedDict where the keys are string tokens, and the values
      are the counts of unique users who have at least one example in the
      training set containing that token in the body text.
    """
    if vocab_size is not None:
        if not isinstance(vocab_size, int):
            raise TypeError(
                f'vocab_size should be None or int, got {type(vocab_size)}.'
            )
        if vocab_size <= 0:
            raise ValueError(f'vocab_size must be positive, got {vocab_size}.')

    path = tf.keras.utils.get_file(
        'stackoverflow.word_count.tar.bz2',
        origin='https://storage.googleapis.com/tff-datasets-public/'
               'stackoverflow.word_count.tar.bz2',
        file_hash=(
            '1dc00256d6e527c54b9756d968118378ae14e6692c0b3b6cad470cdd3f0c519c'
        ),
        hash_algorithm='sha256',
        extract=True,
        archive_format='tar',
        cache_dir=cache_dir,
    )

    word_counts = OrderedDict()
    dir_path = os.path.dirname(path)
    file_path = os.path.join(dir_path, 'stackoverflow.word_count')
    with open(file_path) as f:
        for line in f:
            word, count = line.split()
            word_counts[word] = int(count)
            if vocab_size is not None and len(word_counts) >= vocab_size:
                break
    return word_counts


def _fetch_client_ids(
        database_filepath: str, split_name: Optional[str] = None
) -> Iterator[str]:
    """Fetches the list of client_ids.

    :param database_filepath:
        A path to a SQL database.
    :param split_name:
        An optional split name to filter on. If `None`, all client ids
         are returned.
    :returns:
      An iterator of string client ids.
    """
    connection = sqlite3.connect(database_filepath)
    query = "SELECT DISTINCT client_id FROM client_metadata"
    if split_name is not None:
        query += f" WHERE split_name = '{split_name}'"
    query += ";"
    result = connection.execute(query)
    return map(lambda x: x[0], result)


def fetch_lzma_file(origin: str, filename: str):
    """Fetches a LZMA compressed file and decompresses on the fly."""
    # Read and decompress in approximately megabyte chunks.
    def url_basename(origin: str) -> str:
        origin_path = urllib.parse.urlparse(origin).path
        return origin_path.rsplit('/', maxsplit=1)[-1]

    chunk_size = 2 ** 20
    decompressor = lzma.LZMADecompressor()
    with urllib.request.urlopen(origin) as in_stream, open(
            filename, 'wb') as out_stream:
        length = in_stream.headers.get('content-length')
        if length is not None:
            total_size = int(length)
        else:
            total_size = None
        download_chunk = in_stream.read(chunk_size)
        with tqdm(
            total=total_size, desc=f'Downloading {url_basename(origin)}'
        ) as progbar:
            while download_chunk:
                progbar.update(len(download_chunk))
                out_stream.write(decompressor.decompress(download_chunk))
                download_chunk = in_stream.read(chunk_size)


def query_client_dataset(database_filepath: str, client_id: str,
                         split_name: Optional[str] = None):
    def add_proto_parsing(dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Add parsing of the tf.Example proto to the dataset pipeline."""

        def parse_proto(tensor_proto):
            parse_spec = OrderedDict(
                creation_date=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                score=tf.io.FixedLenFeature(dtype=tf.int64, shape=()),
                tags=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                title=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                tokens=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
                type=tf.io.FixedLenFeature(dtype=tf.string, shape=()),
            )
            parsed_features = tf.io.parse_example(tensor_proto, parse_spec)
            return OrderedDict(
                (key, parsed_features[key]) for key in parse_spec.keys()
            )

        return dataset.map(parse_proto, num_parallel_calls=tf.data.AUTOTUNE)

    query_parts = [
        "SELECT serialized_example_proto FROM examples WHERE client_id = '",
        client_id,
        "'",
    ]
    if split_name is not None:
        query_parts.extend([" and split_name ='", split_name, "'"])
    return add_proto_parsing(tf.data.experimental.SqlDataset(
        driver_name="sqlite",
        data_source_name=database_filepath,
        query=tf.strings.join(query_parts),
        output_types=(tf.string,),
    ))


def get_vocabulary(vocabulary_size: int) -> Dict[str, int]:
    """
    :param vocabulary_size:
        Select the top `vocabulary_size` words to be the vocabulary.
    :returns:
        A dict mapping each word in vocabulary to its embedding id.
        Special tokens:
        * padding - 0
        * unk - 10001
        * bos - 10002
        * eos - 10003
    """
    vocab_list = [PAD] + list(
        load_word_counts(vocab_size=vocabulary_size).keys()) + [UNK, BOS, EOS]
    vocab = defaultdict(lambda: vocab_list.index(UNK))
    vocab.update({t: i for i, t in enumerate(vocab_list)})
    return vocab


def _process_vocabulary(vocabulary_size, h5):
    vocabulary = get_vocabulary(vocabulary_size)
    h5['/metadata/vocabulary_size'] = len(vocabulary)
    h5['/metadata/unk_symbol'] = vocabulary[UNK]
    h5['/metadata/pad_symbol'] = vocabulary[PAD]
    for token, id_ in vocabulary.items():
        # Wrap inside " because of the '.' token.
        h5.create_dataset(
            f'/metadata/vocabulary/"{token}"', data=id_, dtype='int32')
    return vocabulary


def _tokens_to_ids(raw_batch, vocab, max_sequence_length):
    # Encode lists of tokens into a matrix suitable for input to models.
    def tokens_to_word_ids(tokens, vocab):
        return [vocab[word] for word in tokens
                ] + [vocab['PAD']] * (max_sequence_length + 1 - len(tokens))

    to_ret = [tokens_to_word_ids(seq, vocab) for seq in raw_batch]
    return np.array(to_ret, dtype=np.int32)


def _make_worker_fn(database_filepath, partition, vocabulary,
                    max_sequence_length, h5_path, lock):
    def _process_user(user_id):

        # tf Dataset with sentences from user.
        tfdata = query_client_dataset(database_filepath, user_id, partition)

        sentences = []
        for sentence_data in tfdata:
            sentence_tokens = ['BOS'] + sentence_data['tokens'].numpy().decode(
                'UTF-8').split(' ') + ['EOS']
            sentences.append(sentence_tokens)

        trimmed_sentences = [
            sentence[:max_sequence_length + 1] for sentence in sentences
        ]
        token_count = sum(
            [min(len(s), max_sequence_length) for s in trimmed_sentences])
        padded_sentences_ids = _tokens_to_ids(trimmed_sentences, vocabulary,
                                              max_sequence_length)

        with lock:
            with h5py.File(h5_path, 'a') as h5:
                # Store encoded inputs.
                h5.create_dataset(
                    f'/{partition}/{user_id}/inputs',
                    data=padded_sentences_ids[:, :-1])

                # Store encoded targets.
                h5.create_dataset(
                    f'/{partition}/{user_id}/targets',
                    data=padded_sentences_ids[:, 1:])
        return user_id, token_count

    def _worker_fn(work_queue, result_queue):
        # Continuously query for user_ids to process,
        # process them and send results.
        while True:
            user_id = work_queue.get()
            if user_id is None:
                result_queue.put((None, None))
                break
            result_queue.put(_process_user(user_id))

    return _worker_fn


def _collect_num_tokens(results_queue, user_num_tokens, num_processes):
    result_done_count = 0
    while True:
        user_id, token_count = results_queue.get()
        if user_id is None:
            result_done_count += 1
        else:
            # Store token count.
            user_num_tokens[user_id] = token_count
        if result_done_count >= num_processes:
            break
    return user_num_tokens


def dl_preprocess_and_dump_h5(output_dir: str, vocabulary_size: int,
                              max_sequence_length: int, num_processes: int):
    """
    Download and preprocess StackOverflow dataset.

    :param output_dir:
        Directory for all output files, both raw and processed data.
    :param vocabulary_size:
        Size of vocabulary to use, excluding special tokens.
    :param max_sequence_length:
        Trim all sentences to have this maximum length.
    :param num_processes:
        Number of processes to use for parallelizing processing data.
    """
    os.makedirs(output_dir, exist_ok=True)

    database_filepath = os.path.join(output_dir, "stackoverflow.sqlite")
    if not os.path.exists(database_filepath):
        print(f'Downloading StackOverflow data to {output_dir}')
        database_origin = ("https://storage.googleapis.com/tff-datasets-public/"
                           "stackoverflow.sqlite.lzma")
        fetch_lzma_file(origin=database_origin, filename=database_filepath)

    h5_path = os.path.join(output_dir, 'stackoverflow.hdf5')
    # Will overwrite file if exists.
    with h5py.File(h5_path, 'w') as h5:
        vocabulary = _process_vocabulary(vocabulary_size, h5)

    manager = mp.Manager()
    lock = mp.Lock()
    for partition in ['train', 'val', 'test']:
        print(f'Processing users for partition {partition}')
        client_ids = _fetch_client_ids(database_filepath, partition)
        # This is a dict that is shared between main and subprocess.
        user_num_tokens = manager.dict()
        work_queue = mp.Queue()
        results_queue = mp.Queue()
        processes = [
            mp.Process(
                target=_make_worker_fn(database_filepath, partition, vocabulary,
                                       max_sequence_length, h5_path, lock),
                args=(work_queue, results_queue)) for _ in range(num_processes)
        ]
        processes.append(
            mp.Process(
                target=_collect_num_tokens,
                args=(results_queue, user_num_tokens, num_processes)))

        for p in processes:
            p.start()

        for user_id in tqdm(client_ids, 'Queueing work'):
            work_queue.put(user_id)
        for p in processes:
            work_queue.put(None)

        for p in processes:
            p.join()

        if len(user_num_tokens):
            with h5py.File(h5_path, 'a') as h5:
                h5[f'/metadata/num_tokens/{partition}'] = json.dumps(
                    dict(user_num_tokens))


if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        '--output_dir',
        help=('Output directory for the original sqlite '
              'data and the processed hdf5 file.'),
        default='./data/stackoverflow')
    argument_parser.add_argument(
        '--vocab_size',
        type=int,
        default=10000,
        help='Size of vocabulary to use, excluding special tokens.')
    argument_parser.add_argument(
        '--max_sequence_length',
        default=20,
        type=int,
        help='Trim all sentences to have this maximum length.')
    argument_parser.add_argument(
        '--num_processes',
        type=int,
        default=8,
        help='Number of processes to use for parallelizing processing data.')
    arguments = argument_parser.parse_args()

    dl_preprocess_and_dump_h5(arguments.output_dir, arguments.vocab_size,
                              arguments.max_sequence_length,
                              arguments.num_processes)
