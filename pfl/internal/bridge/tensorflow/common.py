# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import dill
import tensorflow as tf

from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops import tensorflow_ops as ops
from pfl.model.tensorflow import TFModel
from pfl.stats import TrainingStatistics

from ..base import CommonFrameworkBridge

_tf_cache = {}


def get_or_make_tf_function(model: TFModel, fn):
    """
    Lookup tf function in cache or create it.
    One graph per model with unique uuid is created.
    """
    id_ = f'{fn.__name__}-{model.uuid}'
    if id_ not in _tf_cache:
        _tf_cache[id_] = fn(model)
    return _tf_cache[id_]


def _detensorify(item):
    if isinstance(item, TrainingStatistics):
        item = item.apply_elementwise(ops.to_numpy)
    return item


def _tensorify(item):
    if isinstance(item, TrainingStatistics):
        item = item.apply_elementwise(ops.to_tensor)
    return item


class TFCommonBridge(CommonFrameworkBridge[TFModel, NNTrainHyperParams]):

    @staticmethod
    def save_state(state: object, path: str):
        state = tf.nest.map_structure(_detensorify, state)
        with open(path, 'wb') as f:
            dill.dump(state, f)

    @staticmethod
    def load_state(path: str):
        with open(path, 'rb') as f:
            state = dill.load(f)
        return tf.nest.map_structure(_tensorify, state)
