
# Copyright © 2023-2024 Apple Inc.
import tensorflow as tf


def _apply_mask(y_true, sample_weight, masked_tokens, dtype):
    sample_weight = tf.ones_like(y_true, dtype) if sample_weight is None else tf.cast(sample_weight, dtype)
    for token in masked_tokens:
        mask = tf.cast(tf.not_equal(y_true, token), dtype)
        sample_weight = sample_weight * mask
    return sample_weight


def _mask_and_flatten_tokens(y_true, y_pred, sample_weight, masked_tokens,
                             dtype):
    sample_weight = _apply_mask(y_true, sample_weight, masked_tokens, dtype)
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, num_classes])
    sample_weight = tf.reshape(sample_weight, [-1])
    return y_true, y_pred, sample_weight


class NumTokensCounter(tf.keras.metrics.Sum):
    """A metric that counts tokens seen after masking."""

    def __init__(self, masked_tokens=None, **kwargs):
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, values, sample_weight=None):
        sample_weight = _apply_mask(values, sample_weight, self._masked_tokens,
                                    self._dtype)
        sample_weight = tf.reshape(sample_weight, [-1])
        super().update_state(sample_weight)


class MaskedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
    """Metric for per-token accuracy with masking."""

    def __init__(self, masked_tokens=None, **kwargs):
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(*_mask_and_flatten_tokens(
            y_true, y_pred, sample_weight, self._masked_tokens, self._dtype))


class MaskedCategoricalCrossentropy(
        tf.keras.metrics.SparseCategoricalCrossentropy):
    """Metric for per-token cross-entropy with masking."""

    def __init__(self, masked_tokens=None, **kwargs):
        self._masked_tokens = masked_tokens or []
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(*_mask_and_flatten_tokens(
            y_true, y_pred, sample_weight, self._masked_tokens, self._dtype))


class Perplexity(MaskedCategoricalCrossentropy):
    """Metric for Perplexity with masking."""

    def result(self):
        return tf.math.exp(super().result())
