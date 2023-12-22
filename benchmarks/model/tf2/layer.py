# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from typing import Optional

import tensorflow as tf

from ..numpy.layer import positional_encoding


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Keras wrapper of positional encoding and embedding layer.
    """

    def __init__(self, vocab_size: int, embedding_size: int,
                 max_sequence_length: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size, mask_zero=True)
        self.pos_encoding = tf.convert_to_tensor(
            positional_encoding(
                length=max_sequence_length, depth=embedding_size))

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of
        # the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class CausalEncoderLayer(tf.keras.layers.Layer):
    """
    Causal encoder of transformer for language modeling. Based on:
    https://github.com/google-research/federated/blob/master/fedopt_guide/stackoverflow_transformer/transformer_models.py
    and the paper `A Field Guide to Federated Optimization` (https://arxiv.org/pdf/2107.06917.pdf),
    where the description of the Transformer model is in Appendix B.2.
    """

    def __init__(self, hidden_size: int, num_heads: int, feedforward_size: int,
                 dropout_rate: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.mha = tf.keras.layers.MultiHeadAttention(
            # key_dim is per head dimension
            key_dim=hidden_size // num_heads,
            num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(feedforward_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    @staticmethod
    def _compute_causal_mask(x: tf.Tensor):
        seq_length = tf.shape(x)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, seq_length, seq_length), tf.bool), -1, 0)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        causal_mask = self._compute_causal_mask(x)
        attn_output = self.mha(x, x, x, attention_mask=causal_mask)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
