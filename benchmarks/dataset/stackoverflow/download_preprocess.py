# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Download the StackOverflow dataset
(https://www.kaggle.com/datasets/stackoverflow/stackoverflow)
using TensoFlow Federated
(https://www.tensorflow.org/federated/api_docs/python/tff/
simulation/datasets/stackoverflow/load_data)
such that experiment setups match that of other frameworks.
Preprocess and save dataset as HDF5 on disk in ML-ready format.
This script is a one-time procedure.
"""
import argparse
from collections import defaultdict
import json
import os
from typing import Dict

import multiprocess as mp
import numpy as np
import h5py
import tensorflow_federated as tff  # pytype: disable=import-error
from tqdm import tqdm

PAD = 'PAD'
UNK = 'UNK'
BOS = 'BOS'
EOS = 'EOS'


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
        tff.simulation.datasets.stackoverflow.load_word_counts(
            vocab_size=vocabulary_size).keys()) + [UNK, BOS, EOS]
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
        h5.create_dataset(f'/metadata/vocabulary/"{token}"',
                          data=id_,
                          dtype='int32')
    return vocabulary


def _tokens_to_ids(raw_batch, vocab, max_sequence_length):
    # Encode lists of tokens into a matrix suitable for input to models.
    def tokens_to_word_ids(tokens, vocab):
        return [vocab[word] for word in tokens
                ] + [vocab['PAD']] * (max_sequence_length + 1 - len(tokens))

    to_ret = [tokens_to_word_ids(seq, vocab) for seq in raw_batch]
    return np.array(to_ret, dtype=np.int32)


def _make_worker_fn(tff_dataset, partition, vocabulary, max_sequence_length,
                    h5_path, lock):

    def _process_user(user_id):

        # tf Dataset with sentences from user.
        tfdata = tff_dataset.create_tf_dataset_for_client(user_id)

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
                h5.create_dataset(f'/{partition}/{user_id}/inputs',
                                  data=padded_sentences_ids[:, :-1])

                # Store encoded targets.
                h5.create_dataset(f'/{partition}/{user_id}/targets',
                                  data=padded_sentences_ids[:, 1:])
        return user_id, token_count

    def _worker_fn(work_queue, result_queue):
        # Contiously query for user_ids to process,
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

    print(f'Downloading StackOverflow data to {output_dir}')
    # Not used, but can be useful to have downloaded.
    _ = tff.simulation.datasets.stackoverflow.load_tag_counts(
        cache_dir=output_dir)
    tff_datasets = tff.simulation.datasets.stackoverflow.load_data(
        cache_dir=output_dir)

    h5_path = os.path.join(output_dir, 'stackoverflow.hdf5')
    # Will overwrite file if exists.
    with h5py.File(h5_path, 'w') as h5:
        vocabulary = _process_vocabulary(vocabulary_size, h5)

    manager = mp.Manager()
    lock = mp.Lock()
    for partition, tff_dataset in zip(['train', 'val', 'test'], tff_datasets):

        print(f'Processing users for partition {partition}')
        # This is a dict that is shared between main and subprocess.
        user_num_tokens = manager.dict()
        work_queue = mp.Queue()
        results_queue = mp.Queue()
        processes = [
            mp.Process(target=_make_worker_fn(tff_dataset, partition,
                                              vocabulary, max_sequence_length,
                                              h5_path, lock),
                       args=(work_queue, results_queue))
            for _ in range(num_processes)
        ]
        processes.append(
            mp.Process(target=_collect_num_tokens,
                       args=(results_queue, user_num_tokens, num_processes)))

        for p in processes:
            p.start()

        for user_id in tqdm(tff_dataset.client_ids, 'Queueing work'):
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
