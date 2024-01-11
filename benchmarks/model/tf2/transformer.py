# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import tensorflow as tf

from model.tf2.layer import PositionalEmbedding, CausalEncoderLayer


def lm_transformer(embedding_size: int,
                   hidden_size: int,
                   num_heads: int,
                   feedforward_size: int,
                   num_transformer_layers: int,
                   vocab_size: int,
                   max_sequence_length: int,
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
    :param dropout_rate:
        Dropout rate applied in the Transformer model.
    :return:
        A Keras model compatible with TF2.
    """

    inputs = tf.keras.layers.Input(shape=(None, ))
    input_embedding = PositionalEmbedding(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        max_sequence_length=max_sequence_length)
    intermediate = input_embedding(inputs)

    if embedding_size != hidden_size:
        intermediate = tf.keras.layers.Dense(hidden_size)(intermediate)

    intermediate = tf.keras.layers.Dropout(dropout_rate)(intermediate)

    for _ in range(num_transformer_layers):
        encoder_layer = CausalEncoderLayer(hidden_size=hidden_size,
                                           num_heads=num_heads,
                                           feedforward_size=feedforward_size,
                                           dropout_rate=dropout_rate)
        intermediate = encoder_layer(intermediate)
    # Dense layer changes dimension from rnn_layer_size to input_embedding_size
    if embedding_size != hidden_size:
        intermediate = tf.keras.layers.Dense(embedding_size)(intermediate)

    logits = tf.matmul(intermediate,
                       input_embedding.embedding.embeddings,
                       transpose_b=True)

    keras_model = tf.keras.Model(inputs=inputs, outputs=logits)
    print(keras_model.summary())
    return keras_model
