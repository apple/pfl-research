# Copyright © 2024 Apple Inc.
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)


def simple_cnn(input_shape: Tuple[int, ...],
               num_outputs: int,
               add_softmax: bool = False) -> tf.keras.models.Model:
    """
    Same model as in https://github.com/apple/pfl-research/blob/main/benchmarks/model/tf2/cnn.py#L21.


    A simple CNN with 2 convolutional layers and one dense hidden layer.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A Keras model compatible with TF2.
    """

    return tf.keras.models.Sequential([
        Conv2D(32,
               kernel_size=(3, 3),
               activation="relu",
               input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_outputs, activation=tf.nn.softmax if add_softmax else None),
    ])
