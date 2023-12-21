# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import tensorflow as tf

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops import tensorflow_ops
from pfl.model.tensorflow import TFModel

from ..base import FedProxFrameworkBridge
from .common import get_or_make_tf_function


def _make_proximal_train_step(model):
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
    def _proximal_train_step(inputs, labels, train_kwargs, global_weights, mu):
        with tf.GradientTape() as tape:
            preds = keras_model(inputs, training=True)
            loss = tf.reduce_mean(keras_model.loss(labels, preds))
            for model_var in keras_model.trainable_variables:
                loss += mu / 2 * tf.norm(
                    model_var - global_weights[model_var.name], 2)**2

        gradients = tape.gradient(loss, keras_model.trainable_variables)
        keras_model.optimizer.apply_gradients(
            zip(gradients, keras_model.trainable_variables))

    return _proximal_train_step


class TFFedProxBridge(FedProxFrameworkBridge[TFModel, NNTrainHyperParams]):
    """
    Concrete implementation of FedProx utilities in TF2, used by
    FedProx algorithm.
    """

    @staticmethod
    def do_proximal_sgd(model: TFModel, user_dataset: AbstractDatasetType,
                        train_params: NNTrainHyperParams, mu: float) -> None:
        global_weights = dict(model.get_parameters().items())
        train_step = get_or_make_tf_function(model, _make_proximal_train_step)
        model.do_multiple_epochs_of(user_dataset,
                                    train_params,
                                    train_step,
                                    global_weights=global_weights,
                                    mu=mu)
