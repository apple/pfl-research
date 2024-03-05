# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock

import numpy as np
import pytest

from pfl.aggregate.base import Backend, get_total_weight_name
from pfl.algorithm.expectation_maximization_gmm import (
    EMGMMHyperParams,
    ExpectationMaximizationGMM,
    make_compute_new_num_components,
)
from pfl.common_types import Population
from pfl.hyperparam import ModelHyperParams
from pfl.internal.distribution.diagonal_gaussian import diagonal_standard_gaussian
from pfl.internal.distribution.mixture import Mixture
from pfl.metrics import Metrics, TrainMetricName
from pfl.model.gaussian_mixture_model import GaussianMixtureModel, GMMHyperParams
from pfl.stats import MappedVectorStatistics

fixed_cohort_size = 100
fixed_val_cohort_size = 99


@pytest.fixture(scope='function')
def mock_gmm_backend(new_event_loop):
    backend = MagicMock(spec=Backend)

    async def mock_async_gather_results(model, training_algorithm,
                                        central_context):

        metrics = Metrics()
        assert isinstance(model, GaussianMixtureModel)
        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        if central_context.population == Population.TRAIN:
            metrics[TrainMetricName('loss',
                                    central_context.population,
                                    after_training=True)] = 0.1

        metrics[TrainMetricName('loss',
                                central_context.population,
                                after_training=False)] = 1337
        metrics[get_total_weight_name(
            central_context.population)] = central_context.cohort_size

        # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        num_components = len(model.components)
        weight = central_context.cohort_size / num_components

        def generate_statistics():
            for (index, (_, component)) in enumerate(model.components):
                yield (f'responsibility-{index}', np.array(weight))
                # Move the mean back to 1.
                mean_value = 1 - component.mean
                yield (f'mean-{index}',
                       (weight * mean_value) / np.sqrt(component.variance))
                # Move the variance back to 1.
                yield (f'variance-{index}', weight *
                       (2 - np.square(component.mean) - component.variance) /
                       component.variance)

        if central_context.population == Population.TRAIN:
            assert central_context.cohort_size == fixed_cohort_size
            model_updates = MappedVectorStatistics(
                dict(generate_statistics()),
                weight=central_context.cohort_size)
        else:
            assert central_context.cohort_size == fixed_val_cohort_size
            model_updates = None

        return model_updates, metrics

    backend.async_gather_results.side_effect = mock_async_gather_results
    return backend


@pytest.mark.parametrize('num_initial_iterations', [0, 1, 3])
@pytest.mark.parametrize('mix_up_interval', [1, 3])
@pytest.mark.parametrize('step_components', [0, 1, 3])
@pytest.mark.parametrize('max_num_components', [10, None])
@pytest.mark.parametrize('fraction_new_components', [0., 0.25, 1.0])
def test_compute_new_num_components(num_initial_iterations, mix_up_interval,
                                    max_num_components, step_components,
                                    fraction_new_components):
    compute_new_num_components = make_compute_new_num_components(
        num_initial_iterations=num_initial_iterations,
        mix_up_interval=mix_up_interval,
        max_num_components=max_num_components,
        step_components=step_components,
        fraction_new_components=fraction_new_components)
    # Simulate multiple iterations of training.
    current_num_components = 1
    num_iterations_since_last_mix_up = 0
    for iteration in range(30):

        new_num_components = compute_new_num_components(
            iteration, num_iterations_since_last_mix_up,
            current_num_components)

        # This needs to be integer.
        assert type(new_num_components) == int

        # Test the behavior.
        if iteration < num_initial_iterations:
            assert new_num_components == current_num_components
        elif num_iterations_since_last_mix_up < mix_up_interval:
            assert new_num_components == current_num_components
        else:
            step_new_components = current_num_components + step_components
            fraction_new_num_components = (
                current_num_components +
                fraction_new_components * current_num_components)
            if max_num_components is not None and (
                    step_new_components >= max_num_components
                    or fraction_new_num_components >= max_num_components):
                assert new_num_components == max_num_components
            else:
                assert (step_new_components <= new_num_components)
                assert (fraction_new_components - 1 < new_num_components)

                assert (new_num_components <= step_new_components) or (
                    new_num_components <= fraction_new_num_components)

        # Perform the operation that ExpectationMaximizationGMM would do.
        if new_num_components != current_num_components:
            num_iterations_since_last_mix_up = 0
        else:
            num_iterations_since_last_mix_up += 1

        current_num_components = new_num_components

    pass


class TestExpectationMaximizationGMM:

    @pytest.mark.parametrize('central_num_iterations', [5, 8, 9, 11, 12])
    def test_run(self, tmpdir, numpy_ops, mock_gmm_backend,
                 central_num_iterations):
        compute_new_num_components = make_compute_new_num_components(
            num_initial_iterations=5,
            mix_up_interval=3,
            max_num_components=64,
            step_components=2)

        algorithm_params = EMGMMHyperParams(
            central_num_iterations=central_num_iterations,
            evaluation_frequency=1,
            val_cohort_size=fixed_val_cohort_size,
            compute_cohort_size=lambda _i, _n: fixed_cohort_size,
            compute_new_num_components=compute_new_num_components)
        algorithm = ExpectationMaximizationGMM()

        model_train_params = GMMHyperParams()
        model_eval_params = ModelHyperParams()
        mixture = Mixture([(1., diagonal_standard_gaussian(2))])
        model = GaussianMixtureModel(
            2, mixture, cached_model_train_params=model_train_params)

        model = algorithm.run(algorithm_params=algorithm_params,
                              backend=mock_gmm_backend,
                              model=model,
                              model_train_params=model_train_params,
                              model_eval_params=model_eval_params)

        # We mix up before these iterations:
        # (5) 1 -> 2 (we asked for 2 extra but we can only duplicate once).
        # (8) 2 -> 4.
        # (11) 4 -> 6.
        # and we've done a final iteration to settle the parameters.
        if central_num_iterations <= 5:
            assert len(model.components) == 1
        elif central_num_iterations <= 8:
            assert len(model.components) == 2
        elif central_num_iterations <= 11:
            assert len(model.components) == 4
        elif central_num_iterations <= 14:
            assert len(model.components) == 6

        for (_, component) in model.components:
            # We always revert to a mean of 1 for the test.
            assert np.allclose(component.mean, np.array([1, 1]))
            assert np.allclose(component.variance, np.array([1, 1]))
