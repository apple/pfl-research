# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest

from pfl.context import CentralContext, UserContext
from pfl.metrics import Metrics, StringMetricName
from pfl.postprocessor.quantization import QuantizationPostprocessor
from pfl.stats import MappedVectorStatistics


class TestQuantizationPostprocessor:

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            QuantizationPostprocessor(nbits=1)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(nbits=9)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(nbits=4.0)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(nbits=True)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(block_size=0)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(block_size=-2)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(block_size=64.0)
        with pytest.raises(ValueError):
            QuantizationPostprocessor(min_elements=-1)

    def test_postprocess_one_user_int4(self):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((10, 20)).astype(np.float32)
        w2 = rng.standard_normal((5, 10)).astype(np.float32)

        stats = MappedVectorStatistics({'layer1': w1.copy(), 'layer2': w2.copy()})
        postprocessor = QuantizationPostprocessor(nbits=4, block_size=32, min_elements=32)
        context = UserContext(num_datapoints=10, seed=123)

        processed_stats, metrics = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        assert processed_stats['layer1'].shape == w1.shape
        assert processed_stats['layer2'].shape == w2.shape
        assert processed_stats['layer1'].dtype == np.float32

        mse1 = np.mean((w1 - processed_stats['layer1']) ** 2)
        mse2 = np.mean((w2 - processed_stats['layer2']) ** 2)
        assert mse1 < 0.05
        assert mse2 < 0.05

        assert metrics[StringMetricName("quantization | nbits")] == 4
        assert StringMetricName("quantization | mse") in metrics

    def test_postprocess_one_user_int8(self):
        rng = np.random.default_rng(7)
        w = rng.standard_normal((16, 16)).astype(np.float32)

        stats = MappedVectorStatistics({'w': w.copy()})
        postprocessor = QuantizationPostprocessor(nbits=8, block_size=64, min_elements=64)
        context = UserContext(num_datapoints=10, seed=None)

        processed_stats, metrics = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        mse = np.mean((w - processed_stats['w']) ** 2)
        assert mse < 0.001
        assert metrics[StringMetricName("quantization | nbits")] == 8

    def test_min_elements_cutoff(self):
        small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        stats = MappedVectorStatistics({'small': small.copy()})
        postprocessor = QuantizationPostprocessor(nbits=4, min_elements=64)
        context = UserContext(num_datapoints=5, seed=None)

        processed_stats, _ = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        np.testing.assert_array_equal(processed_stats['small'], small)

    def test_non_floating_types_untouched(self):
        int_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        stats = MappedVectorStatistics({'steps': int_arr.copy()})
        postprocessor = QuantizationPostprocessor(nbits=4, min_elements=4)
        context = UserContext(num_datapoints=1, seed=None)

        processed_stats, _ = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        np.testing.assert_array_equal(processed_stats['steps'], int_arr)
        assert processed_stats['steps'].dtype == np.int64

    def test_zero_tensors_handled(self):
        zeros = np.zeros((8, 8), dtype=np.float32)
        stats = MappedVectorStatistics({'w': zeros.copy()})
        postprocessor = QuantizationPostprocessor(nbits=4, block_size=16, min_elements=16)
        context = UserContext(num_datapoints=1, seed=None)

        processed_stats, _ = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        np.testing.assert_array_equal(processed_stats['w'], zeros)

    def test_non_finite_rejection(self):
        with_nan = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        stats = MappedVectorStatistics({'w': with_nan})
        postprocessor = QuantizationPostprocessor(nbits=4, min_elements=2)
        context = UserContext(num_datapoints=1, seed=None)

        with pytest.raises(ValueError, match="Input array contains non-finite values"):
            postprocessor.postprocess_one_user(stats=stats, user_context=context)

    def test_stochastic_quantization_bounded(self):
        rng = np.random.default_rng(99)
        w = rng.standard_normal((10, 10)).astype(np.float32)
        stats = MappedVectorStatistics({'w': w.copy()})
        postprocessor = QuantizationPostprocessor(nbits=4, stochastic=True, block_size=32, min_elements=32)
        context = UserContext(num_datapoints=1, seed=42)

        processed_stats, _ = postprocessor.postprocess_one_user(
            stats=stats, user_context=context)

        diff = np.abs(w - processed_stats['w'])
        assert np.all(diff < 2.0)

    def test_metrics_only_hooks_defaults(self, user_context, central_context):
        postprocessor = QuantizationPostprocessor(nbits=4)
        u_metrics = postprocessor.postprocess_one_user_metrics(user_context=user_context)
        s_metrics = postprocessor.postprocess_server_metrics(
            central_context=central_context, aggregate_metrics=Metrics())

        assert isinstance(u_metrics, Metrics)
        assert len(u_metrics) == 0
        assert isinstance(s_metrics, Metrics)
        assert len(s_metrics) == 0
