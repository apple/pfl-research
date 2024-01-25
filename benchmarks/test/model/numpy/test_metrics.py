# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest
from model.numpy.metrics import AUC, PRAUC, ROCAUC, AveragedPrecision
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from pfl.internal.ops import get_tf_major_version

_seed = 42


def pr_auc_score(y_true, y_score, average):
    if average == "micro":
        precision, recall, _ = precision_recall_curve(
            y_true=y_true.ravel(), probas_pred=y_score.ravel())
        return auc(recall, precision)
    elif average == "macro":
        num_classes = y_score.shape[1]
        return np.mean([
            pr_auc_score(y_true[:, c], y_score[:, c], average="micro")
            for c in range(num_classes)
        ])
    else:
        raise ValueError(average)


def _get_random_probabilities():
    batch_size = 512
    num_labels = 100
    return np.random.uniform(size=(batch_size, num_labels))


def _get_bucket_auc(auc_type):
    if auc_type == "roc":
        return ROCAUC
    elif auc_type == "pr":
        return PRAUC
    elif auc_type == "ap":
        return AveragedPrecision
    else:
        raise ValueError(auc_type)


def _get_sklearn_auc(auc_type):
    if auc_type == "roc":
        return roc_auc_score
    elif auc_type == "pr":
        return pr_auc_score
    elif auc_type == "ap":
        return average_precision_score
    else:
        raise ValueError(auc_type)


def _get_keras_auc(auc_type, multi_label):
    from tensorflow.keras.metrics import AUC as KerasAUC  # type: ignore
    keras_roc_auc = lambda multi_label: KerasAUC(multi_label=multi_label,
                                                 curve="roc")
    keras_pr_auc = lambda multi_label: KerasAUC(multi_label=multi_label,
                                                curve="pr")

    if auc_type == "roc":
        return keras_roc_auc(multi_label)
    elif auc_type == "pr":
        return keras_pr_auc(multi_label)
    else:
        raise ValueError(auc_type)


class TestAUC:

    @pytest.mark.parametrize('auc_type', ["roc", "pr", "ap"])
    @pytest.mark.parametrize('multi_label', [True, False])
    def test_auc_aggregation(self, auc_type, multi_label):
        np.random.seed(_seed)
        num_workers = 10
        aggregated_auc = None
        aggregated_y_score = []
        aggregated_y_true = []
        for _ in range(num_workers):
            y_score = _get_random_probabilities()
            y_true = np.round(_get_random_probabilities()).astype(int)
            worker_auc = _get_bucket_auc(auc_type)(y_true=y_true,
                                                   y_pred=y_score,
                                                   multi_label=multi_label)
            if aggregated_auc is None:
                aggregated_auc = worker_auc
            else:
                aggregated_auc += worker_auc
            aggregated_y_score.append(y_score)
            aggregated_y_true.append(y_true)

        aggregated_y_score = np.vstack(aggregated_y_score)
        aggregated_y_true = np.vstack(aggregated_y_true)
        expected_auc = _get_bucket_auc(auc_type)(
            y_true=aggregated_y_true,
            y_pred=aggregated_y_score,
            multi_label=multi_label).overall_value
        assert isinstance(aggregated_auc, AUC)
        aggregated_auc = aggregated_auc.overall_value
        assert np.isclose(expected_auc, aggregated_auc), (
            f"Expected {auc_type} AUC value: {expected_auc},"
            f" Aggregated {auc_type} AUC value: {aggregated_auc}")

    @pytest.mark.parametrize('auc_type', ["roc", "pr", "ap"])
    @pytest.mark.parametrize('multi_label', [True, False])
    def test_auc_against_sklearn(self, auc_type, multi_label):
        np.random.seed(_seed)
        num_test_cases = 10
        for _ in range(num_test_cases):
            y_score = _get_random_probabilities()
            y_true_random = np.round(_get_random_probabilities()).astype(int)
            y_true_optimal = np.round(y_score).astype(int)
            for y_true in [y_true_random, y_true_optimal]:
                bucket_value = _get_bucket_auc(auc_type)(
                    y_true=y_true, y_pred=y_score,
                    multi_label=multi_label).overall_value
                sklearn_value = _get_sklearn_auc(auc_type)(
                    y_true=y_true,
                    y_score=y_score,
                    average="macro" if multi_label else "micro")
                assert np.isclose(bucket_value, sklearn_value, atol=5e-3), (
                    f"Our {auc_type} AUC value: {bucket_value},"
                    f" sklearn {auc_type} AUC value: {sklearn_value}")

    @pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
    @pytest.mark.parametrize('auc_type', ["roc", "pr"])
    @pytest.mark.parametrize('multi_label', [True, False])
    def test_auc_against_keras(self, auc_type, multi_label):
        np.random.seed(_seed)
        num_test_cases = 10
        for _ in range(num_test_cases):
            y_score = _get_random_probabilities()
            y_true_random = np.round(_get_random_probabilities()).astype(int)
            y_true_optimal = np.round(y_score).astype(int)
            for y_true in [y_true_random, y_true_optimal]:
                bucket_value = _get_bucket_auc(auc_type)(
                    y_true=y_true, y_pred=y_score,
                    multi_label=multi_label).overall_value
                keras_auc = _get_keras_auc(auc_type, multi_label)
                keras_auc.update_state(y_true=y_true, y_pred=y_score)
                keras_value = keras_auc.result().numpy()
                assert np.isclose(bucket_value, keras_value), (
                    f"Our {auc_type} AUC value: {bucket_value}, "
                    f"Keras {auc_type} AUC value: {keras_value}")
