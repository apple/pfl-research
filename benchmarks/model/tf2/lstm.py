# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import tensorflow as tf


def lm_lstm(embedding_size: int, num_cell_states: int, num_lstm_layers: int,
            vocab_size: int):
    """
    Defines a stacked LSTM model with shared input and output embeddings.
    This model is from
    https://github.com/google-research/federated/blob/master/utils/models/stackoverflow_models.py, # pylint: disable=line-too-long
    as used in McMahan et al. (2018) and Konecny et al. (2016). See readme
    for the papers. Confirmed to work with TensorFlow >=2.2.0,<3.

    :param embedding_size:
        Number of parameters in each word embedding vector.
    :param num_cell_states:
        Size of the hidden state of the LSTM
    :param num_lstm_layers:
        Number of LSTM layers stacked on eachother.
    :param vocab_size:
        The size of the input one-hot encodings.
    :return:
        A Keras model compatible with TF2.
    """

    inputs = tf.keras.layers.Input(shape=(None, ))
    # Embedding layer automatically masks padding for consecutive layers.
    input_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
    intermediate = input_embedding(inputs)

    for _ in range(num_lstm_layers):
        layer = tf.keras.layers.LSTM(num_cell_states, return_sequences=True)
        intermediate = layer(intermediate)
    # Dense layer changes dimension from rnn_layer_size to input_embedding_size
    if embedding_size != num_cell_states:
        intermediate = tf.keras.layers.Dense(embedding_size)(intermediate)

    logits = tf.matmul(
        intermediate, input_embedding.embeddings, transpose_b=True)

    keras_model = tf.keras.Model(inputs=inputs, outputs=logits)
    print(keras_model.summary())
    return keras_model
