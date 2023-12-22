# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import math
import torch
import torch.nn as nn

from .layer import PositionalEncoding
from .lstm import LMBase


class LMTransformer(LMBase):
    def __init__(self, embedding_size: int, hidden_size: int, num_heads: int,
                 feedforward_size: int, num_transformer_layers: int,
                 dropout_rate: float, vocab_size: int,
                 max_sequence_length: int, pad_symbol: int, unk_symbol: int):
        super().__init__(pad_symbol, unk_symbol)
        self._embedding_size = embedding_size
        self._embeddings = nn.Embedding(
            vocab_size, embedding_size, padding_idx=pad_symbol)
        self._positional_encoder = PositionalEncoding(embedding_size,
                                                      max_sequence_length)
        encoder_layers = nn.TransformerEncoderLayer(
            hidden_size,
            num_heads,
            feedforward_size,
            dropout_rate,
            batch_first=True)
        self._transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_transformer_layers)

        if embedding_size != hidden_size:
            self._proj_in = nn.Linear(embedding_size, hidden_size)
            self._proj_out = nn.Linear(hidden_size, embedding_size)
        else:
            self._proj_in = self._proj_out = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.05
        self._embeddings.weight.data.uniform_(-initrange, initrange)
        for proj in [self._proj_in, self._proj_out]:
            if isinstance(proj, nn.Linear):
                proj.bias.data.zero_()
                nn.init.xavier_uniform_(proj.weight.data)

    @staticmethod
    def _compute_causal_mask(x: torch.Tensor):
        seq_length = x.size(1)
        # 1 - lower triangular boolean matrix (True -> not attending)
        return ~torch.tril(
            torch.ones(
                (seq_length, seq_length), dtype=torch.bool, device=x.device))

    def forward(self, inputs):
        casual_mask = self._compute_causal_mask(inputs)
        x = self._embeddings(inputs) * math.sqrt(self._embedding_size)
        x = self._positional_encoder(x)
        x = self._proj_in(x)
        x = self._transformer_encoder(x, casual_mask)
        x = self._proj_out(x)
        logits = x @ self._embeddings.weight.t()
        return logits


def lm_transformer(embedding_size: int,
                   hidden_size: int,
                   num_heads: int,
                   feedforward_size: int,
                   num_transformer_layers: int,
                   vocab_size: int,
                   max_sequence_length: int,
                   pad_symbol: int,
                   unk_symbol: int,
                   dropout_rate: float = 0.1):
    """
    Defines a Transformer model with shared input and output embeddings.
    This model is from
    https://github.com/google-research/federated/blob/master/fedopt_guide/stackoverflow_transformer/transformer_models.py
    as used in Wang et al. (2021). See readme for the papers.

    :param embedding_size:
        Number of parameters in each word embedding vector.
    :param hidden_size:
        Size of the hidden state of the Transformer.
    :param num_heads:
        Number of heads in the multi-head attention layer.
    :param feedforward_size:
        Size of the feed forward layer in the Transformer.
    :param num_transformer_layers:
        Number of Transformer layers stacked on eachother.
    :param vocab_size:
        The size of the input one-hot encodings.
    :param max_sequence_length:
        Sequence length to decide the size of positional encoding.
    :param pad_symbol:
        Vocabulary index for PAD symbol.
    :param unk_symbol:
        Vocabulary index for UNK symbol.
    :param dropout_rate:
        Dropout rate applied in the Transformer model.
    :return:
        A PyTorch Transformer model.
    """
    return LMTransformer(embedding_size, hidden_size, num_heads,
                         feedforward_size, num_transformer_layers,
                         dropout_rate, vocab_size, max_sequence_length,
                         pad_symbol, unk_symbol)
