# Copyright © 2023-2024 Apple Inc.
from typing import Tuple

import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)

# yapf: disable
from tensorflow_addons.layers.normalizations import GroupNormalization  # type: ignore


def simple_cnn(input_shape: Tuple[int, ...],
               num_outputs: int) -> tf.keras.models.Model:
    """
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
               activation='relu',
               input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_outputs, activation=None),
    ])


def _conv_norm(x, filters, kernel_size, weight_decay=1e-4, strides=(1, 1)):
    """
    Convolution + normalization layer with parameters used in ResNets.
    Batch norm is replaced by Group norm because it works better for federated
    learning.
    """
    layer = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    layer = GroupNormalization(axis=3)(layer)
    return layer


def _conv_norm_relu(x,
                    filters,
                    kernel_size,
                    weight_decay=1e-4,
                    strides=(1, 1)):
    layer = _conv_norm(x, filters, kernel_size, weight_decay, strides)
    layer = tf.nn.relu(layer)
    return layer


def _residual_block(x,
                    filters,
                    kernel_size,
                    weight_decay=1e-4,
                    downsample=True):
    """ Residual block used in ResNets. """
    if downsample:
        residual_x = _conv_norm(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = _conv_norm_relu(
        x,
        filters=filters,
        kernel_size=kernel_size,
        weight_decay=weight_decay,
        strides=stride,
    )
    residual = _conv_norm(
        residual,
        filters=filters,
        kernel_size=kernel_size,
        weight_decay=weight_decay,
        strides=1,
    )
    out = tf.keras.layers.add([residual_x, residual])
    out = tf.nn.relu(out)
    return out


def resnet18(input_shape: Tuple[int, ...],
             num_outputs: int) -> tf.keras.models.Model:
    """
    Define a ResNet18 model with group normalization.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A Keras model compatible with TF2.
    """
    input_ = Input(shape=input_shape)
    x = _conv_norm_relu(input_, filters=64, kernel_size=(7, 7), strides=(2, 2))
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Residual block 1
    x = _residual_block(x, filters=64, kernel_size=(3, 3), downsample=False)
    x = _residual_block(x, filters=64, kernel_size=(3, 3), downsample=False)
    # Residual block 2
    x = _residual_block(x, filters=128, kernel_size=(3, 3), downsample=True)
    x = _residual_block(x, filters=128, kernel_size=(3, 3), downsample=False)
    # Residual block 3
    x = _residual_block(x, filters=256, kernel_size=(3, 3), downsample=True)
    x = _residual_block(x, filters=256, kernel_size=(3, 3), downsample=False)
    # Residual block 4
    x = _residual_block(x, filters=512, kernel_size=(3, 3), downsample=True)
    x = _residual_block(x, filters=512, kernel_size=(3, 3), downsample=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(
        num_outputs,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return tf.keras.models.Model(input_, x, name='ResNet18')
