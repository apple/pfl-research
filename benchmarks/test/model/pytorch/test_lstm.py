# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest

from pfl.internal.ops import get_pytorch_major_version

if get_pytorch_major_version():
    from model.pytorch import lm_lstm
    import torch

_PAD_SYMBOL = 0
_UNK_SYMBOL = 1


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
class TestPyTorchCNN:

    @pytest.mark.parametrize(
        'embedding_size,num_cell_states,num_lstm_layers,vocab_size,sequence_length',
        [(64, 64, 1, 1000, 10), (128, 128, 2, 5000, 20)])
    def test_output_shape(self, embedding_size: int, num_cell_states: int,
                          num_lstm_layers: int, vocab_size: int,
                          sequence_length: int):
        pytorch_model = lm_lstm(embedding_size, num_cell_states,
                                num_lstm_layers, vocab_size, _PAD_SYMBOL,
                                _UNK_SYMBOL)
        batch_size = 3
        inputs = np.random.randint(0,
                                   vocab_size,
                                   size=(batch_size, sequence_length))
        logits = pytorch_model(torch.from_numpy(inputs))
        assert logits.shape == (batch_size, sequence_length, vocab_size)

    @pytest.mark.parametrize(
        'embedding_size,num_cell_states,num_lstm_layers,num_lstm_layers', [
            (64, 64, 1, 1000),
            (128, 128, 2, 5000),
        ])
    def test_num_parameters(self, embedding_size: int, num_cell_states: int,
                            num_lstm_layers: int, vocab_size: int):
        pytorch_model = lm_lstm(embedding_size, num_cell_states,
                                num_lstm_layers, vocab_size, _PAD_SYMBOL,
                                _UNK_SYMBOL)
        # Embedding layer of shape `(vocab_size, embedding_size)`
        num_parameters = embedding_size * vocab_size
        for idx, _ in enumerate(range(num_lstm_layers)):
            input_size = embedding_size if idx == 0 else num_cell_states
            # (W_ii|W_if|W_ig|W_io)`, of shape `(4*num_cell_states, input_size)`
            num_parameters += input_size * num_cell_states * 4
            # (W_hi|W_hf|W_hg|W_ho)`, of shape `(4*num_cell_states, hidden_size)
            num_parameters += num_cell_states * num_cell_states * 4
            # (b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)
            num_parameters += num_cell_states * 4
            # (b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)
            num_parameters += num_cell_states * 4
        # Projection layer if dimension does not match
        if embedding_size != num_cell_states:
            # Weight shape `(num_cell_states, embedding_size)` + bias shape
            num_parameters += num_cell_states * embedding_size + embedding_size
        pytorch_model_num_parameters = sum(
            [np.prod(tuple(var.shape)) for var in pytorch_model.parameters()])
        assert num_parameters == pytorch_model_num_parameters
