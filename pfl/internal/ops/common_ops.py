# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import os
from packaging.version import Version

from .selector import get_framework_module as get_ops


def all_reduce_metrics(metrics):
    """
    Performs all reduce between workers on a `Metrics` object.
    If the current instance is not connected to a cluster, this will
    return the identity.

    When one worker calls this method, it will block until `all_reduce_metrics`
    has been called on all other worker instances as well.

    :param metrics:
        A `Metrics` object that contains metrics from training with the data
        that was assigned for the local worker.
    :returns:
        A `Metrics` object where all its elements has been summed across all
        workers.
        The returned `Metrics` now contains the same values for each worker.
    """
    metric_value_vectors = [
        get_ops().to_tensor(v) for v in metrics.to_vectors()
    ]
    reduced_metric_value_vectors = [
        get_ops().to_numpy(v)
        for v in get_ops().distributed.all_reduce(metric_value_vectors)
    ]
    return metrics.from_vectors(reduced_metric_value_vectors)


def all_reduce_metrics_and_stats(stats, metrics):
    metadata, weights = stats.get_weights()
    metric_value_vectors = [
        get_ops().to_tensor(v) for v in metrics.to_vectors()
    ]

    reduced_data = get_ops().distributed.all_reduce(
        [metadata, *metric_value_vectors, *list(weights)])

    reduced_metadata = reduced_data[0]
    reduced_metric_vectors = reduced_data[1:1 + len(metric_value_vectors)]
    reduced_weights = reduced_data[1 + len(metric_value_vectors):]
    model_update = stats.from_weights(reduced_metadata, reduced_weights)

    reduced_metric_value_vectors = [
        get_ops().to_numpy(v) for v in reduced_metric_vectors
    ]
    reduced_metrics = metrics.from_vectors(reduced_metric_value_vectors)
    return model_update, reduced_metrics


def get_tf_major_version() -> int:
    """
    :returns:
        The major version of the TensorFlow package installed.
        ``0`` if TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ModuleNotFoundError:
        # No TensorFlow package installed.
        return 0

    # Return major version number
    return int(Version(tf.__version__).major)


def get_pytorch_major_version() -> int:
    """
    :returns:
        The major version of the PyTorch package installed.
        ``0`` if PyTorch is not installed.
    """
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        # No TensorFlow package installed.
        return 0

    torch_version = Version(torch.__version__)
    try:
        # Return major version number
        return int(torch_version.major)
    except AttributeError:
        # This may happen when building documentation.
        return 0


def check_pfl_tree_installed() -> bool:
    """
    :returns:
        True if pfl is set up to train trees. False otherwise.
    """
    try:
        import xgboost  # pylint: disable=unused-import
        return True
    except ModuleNotFoundError:
        return False


def is_pytest_running():
    """
    :returns:
        `True` if pytest is currently running.
    """
    return 'PYTEST_CURRENT_TEST' in os.environ
