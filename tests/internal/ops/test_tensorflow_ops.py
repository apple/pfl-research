# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest

from pfl.internal.ops import get_tf_major_version

if get_tf_major_version() > 1:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import backend as K

    from pfl.internal.ops import tensorflow_ops  # pylint: disable=ungrouped-imports


@pytest.fixture
def labels1():
    return np.array([[0, 1], [0, 1]])


@pytest.fixture
def labels2():
    return np.array([[1, 1], [1, 1]])


@pytest.fixture
def preds():
    return np.array([[1, 1], [1, 1]])


@pytest.fixture
def keras_metric():
    return tf.keras.metrics.MeanAbsoluteError()  # pytype: disable=module-attr


@pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
class TestKerasMetricValue:

    def test_add(self, keras_metric, labels1, labels2, preds):

        m1 = tensorflow_ops.KerasMetricValue(keras_metric, labels1, preds)
        m2 = tensorflow_ops.KerasMetricValue(keras_metric, labels2, preds)
        m_sum = m1 + m2
        assert m_sum.overall_value == 2 / 8
        assert m1 is not m_sum
        assert m2 is not m_sum

    def test_equality(self, keras_metric, labels1, labels2, preds):
        m1 = tensorflow_ops.KerasMetricValue(keras_metric, labels1, preds)
        m2 = tensorflow_ops.KerasMetricValue(keras_metric, labels1, preds)
        m3 = tensorflow_ops.KerasMetricValue(keras_metric, labels2, preds)
        assert m1 == m2
        assert m1 != m3

    def test_add_morphism(self, keras_metric, labels1, labels2, preds):
        m1 = tensorflow_ops.KerasMetricValue(keras_metric, labels1, preds)
        m2 = tensorflow_ops.KerasMetricValue(keras_metric, labels2, preds)
        vector = (m1.to_vector() + m2.to_vector())
        total = m1.from_vector(vector)
        assert total == (m1 + m2)
        assert total is not m1
        assert total is not m2
