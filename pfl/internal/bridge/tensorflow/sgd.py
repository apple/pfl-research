# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import tensorflow as tf

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops import tensorflow_ops
from pfl.model.tensorflow import TFModel

from ..base import SGDFrameworkBridge
from .common import get_or_make_tf_function


def _make_train_step(model):
    # The least complicated way to make tf.function functions that include
    # the model as a global variable, and being able to re-use that graph
    # as long as the same model is used. Inputing keras_model as argument
    # to the graph fn "works" in simple cases, but is not very stable.
    # If the model object changes, e.g. a variable in the optimizer changes,
    # things go wrong.
    # Also, large inputs to tf.function, e.g. a dict of all trainable
    # variables, introduces large overhead in calculating cache hash by
    # tf.function.
    keras_model = model.keras_model

    @tensorflow_ops.tf_function(experimental_relax_shapes=True)
    def _train_step(inputs, labels, train_kwargs, max_grad_norm):  # pylint: disable=arguments-differ
        with tf.GradientTape() as tape:
            preds = keras_model(inputs, training=True)
            # TODO: The mean is used because that's what model.fit do.
            # We should also have to option of training on the summed loss
            # rdar://70432095.
            loss = tf.reduce_mean(keras_model.loss(labels, preds))
        gradients = tape.gradient(loss, keras_model.trainable_variables)
        if max_grad_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        keras_model.optimizer.apply_gradients(
            zip(gradients, keras_model.trainable_variables))

    return _train_step


class TFSGDBridge(SGDFrameworkBridge[TFModel, NNTrainHyperParams]):
    """
    Concrete TF implementations of utils for stochastic gradient
    descent.
    """

    @staticmethod
    def do_sgd(model: TFModel, user_dataset: AbstractDatasetType,
               train_params: NNTrainHyperParams) -> None:
        train_step = get_or_make_tf_function(model, _make_train_step)
        model.do_multiple_epochs_of(
            user_dataset,
            train_params,
            train_step,
            max_grad_norm=train_params.local_max_grad_norm)
