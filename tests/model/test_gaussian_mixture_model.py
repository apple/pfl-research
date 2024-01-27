# Copyright © 2023-2024 Apple Inc.

import math
import os
import pickle
from tempfile import TemporaryDirectory
from typing import Callable, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from pfl.data.dataset import AbstractDataset, Dataset
from pfl.internal.distribution import DiagonalGaussian, Mixture, diagonal_standard_gaussian
from pfl.internal.distribution.distribution import any_sum
from pfl.metrics import MetricName, Metrics, MetricValue, StringMetricName, Weighted
from pfl.model.gaussian_mixture_model import GaussianMixtureModel, GMMHyperParams
from pfl.stats import MappedVectorStatistics


def unscented_points_standard(num_dimensions):
    """
    Return a small number of data points with sample mean 0 and sample variance
    1.
    This is from the "unscented transform".
    """

    # These are points with the right shape, but not the right scale:
    points = [
        np.identity(num_dimensions),
        -np.identity(num_dimensions),
    ]

    # If you were to compute the variance of "points", this would be the answer:
    variance = 2 / (num_dimensions * 2)

    # Scale to compensate for this.
    return np.concatenate(points, axis=0) / math.sqrt(variance)


def test_unscented_points_standard():
    for num_dimensions in [1, 2, 3, 4, 10]:
        points = unscented_points_standard(num_dimensions)

        mean = np.mean(points, axis=0)
        variance = np.mean(np.square(points), axis=0) - mean

        assert np.isclose(mean, 0).all()
        assert np.isclose(variance, 1).all()


def unscented_points(num_dimensions, mean, variance):
    return mean + np.sqrt(variance) * unscented_points_standard(num_dimensions)


def unscented_user_data_set(num_dimensions, mean, variance):
    return Dataset(unscented_points(num_dimensions, mean, variance))


def test_construction_default():
    unit_gaussian_model = GaussianMixtureModel(3)
    ((weight, unit_gaussian), ) = unit_gaussian_model.components
    assert weight == 1
    assert unit_gaussian.num_dimensions == 3
    assert (unit_gaussian.mean == 0).all()
    assert (unit_gaussian.variance == 1).all()


def test_construction():
    gaussian = DiagonalGaussian([1, 2], [4, 5])
    gaussian_model = GaussianMixtureModel(2, Mixture([(1, gaussian)]))

    ((weight, gaussian_1), ) = gaussian_model.components
    assert weight == 1
    assert gaussian_1 is gaussian


@pytest.fixture
def gmm_params():
    return Mock(spec=GMMHyperParams,
                variance_scale=.2,
                responsibility_scale=4,
                variance_floor_fraction=0.01,
                minimum_component_weight=0.01)


def test_construction_cached_model_train_params(gmm_params):
    gaussian_model = GaussianMixtureModel(2,
                                          cached_model_train_params=gmm_params)
    # pylint: disable=protected-access
    assert gaussian_model._cached_model_train_params == gmm_params
    # pylint: enable=protected-access


def test_evaluate():
    gaussian_1 = DiagonalGaussian([1, 2], [4, 5])
    gaussian_2 = DiagonalGaussian([2, 3], [7, 8])
    underlying_model = Mixture([(1, gaussian_1), (.5, gaussian_2)])
    mixture_model = GaussianMixtureModel(2, underlying_model)

    data = [np.array([4., 3.5]), np.array([2.5, 3.])]
    joint_density = (underlying_model.density(data[0]) *
                     underlying_model.density(data[1]))

    mock_dataset = Mock(AbstractDataset, raw_data=data)

    def format_name(description):
        return StringMetricName(description)

    metrics = mixture_model.evaluate(mock_dataset,
                                     name_formatting_fn=format_name)
    metric = metrics[format_name('log-likelihood')]
    assert isinstance(metric, MetricValue)
    assert metric.overall_value == pytest.approx(joint_density.log_value / 2)
    # Check that the overall value from multiple individuals behaves as
    # expected.
    assert (metric + metric).overall_value == pytest.approx(
        joint_density.log_value / 2)


def test_save():
    gaussian_1 = DiagonalGaussian([1, 2], [4, 5])
    gaussian_model = GaussianMixtureModel(2, Mixture([(1, gaussian_1)]))

    with TemporaryDirectory() as path:
        gaussian_model.save(path)
        with open(os.path.join(path, 'trained_gmm.pickle'), 'rb') as f:
            model = pickle.load(f)
            assert isinstance(model, Mixture)
            (weight, component), = model.components
            assert weight == 1
            assert np.array_equal(component.mean, gaussian_1.mean)
            assert np.array_equal(component.variance, gaussian_1.variance)


@pytest.mark.parametrize('initial_mean, initial_variance', [([0], [1]),
                                                            ([0], [4]),
                                                            ([2], [4]),
                                                            ([7, 8], [2, 3])])
@pytest.mark.parametrize('variance_scale, responsibility_scale', [(1., 1.),
                                                                  (2., .5),
                                                                  (.1, 3)])
def test_statistics_round_trip_single_component(initial_mean, initial_variance,
                                                variance_scale,
                                                responsibility_scale):
    """
    Test that whatever transformations due to the old Gaussian's parameters and
    the scaling factors, the correct Gaussian parameters are found.
    """
    single_mixture = Mixture([(1,
                               DiagonalGaussian(initial_mean,
                                                initial_variance))])

    num_dimensions = len(initial_mean)
    model = GaussianMixtureModel(num_dimensions, single_mixture)

    for mean, variance in [(0, 1), (0, 3), (-3, 1), (5, 7)]:

        points = unscented_points(num_dimensions, mean, variance)

        variance_floor = 1e-7

        # pylint: disable=protected-access
        _likelihood, statistics = model.get_mixture_statistics(
            points, variance_scale, responsibility_scale)

        ((weight, gaussian), ) = model._maximum_likelihood_gaussians(
            statistics, variance_scale, responsibility_scale, variance_floor)
        # pylint: enable=protected-access

        assert weight == pytest.approx(1)
        assert gaussian.mean == pytest.approx(mean)
        assert gaussian.variance == pytest.approx(variance)


@pytest.mark.parametrize('variance_scale, responsibility_scale', [(1., 1.),
                                                                  (2., .5),
                                                                  (.1, 3)])
def test_statistics_round_trip_two_components(variance_scale,
                                              responsibility_scale):
    mixture = Mixture([
        (1, DiagonalGaussian([-10], [3])),
        (2, DiagonalGaussian([+10], [2])),
    ])

    num_dimensions = 1
    model = GaussianMixtureModel(num_dimensions, mixture)

    # Put the points far enough away that responsibilities become 0-1.
    mean_1 = -15
    variance_1 = 4
    mean_2 = +18
    variance_2 = 5

    points = np.concatenate(
        (unscented_points(num_dimensions, mean_1, variance_1),
         unscented_points(num_dimensions, mean_2, variance_2),
         unscented_points(num_dimensions, mean_2, variance_2)),
        axis=0)

    variance_floor = 1e-7

    # pylint: disable=protected-access
    _likelihood, statistics = model.get_mixture_statistics(
        points, variance_scale, responsibility_scale)

    ((weight_1, gaussian_1),
     (weight_2, gaussian_2)) = model._maximum_likelihood_gaussians(
         statistics, variance_scale, responsibility_scale, variance_floor)
    # pylint: enable=protected-access

    assert weight_1 == pytest.approx(1 / 3)
    assert gaussian_1.mean == pytest.approx(mean_1)
    assert gaussian_1.variance == pytest.approx(variance_1)

    assert weight_2 == pytest.approx(2 / 3)
    assert gaussian_2.mean == pytest.approx(mean_2)
    assert gaussian_2.variance == pytest.approx(variance_2)


@pytest.fixture
def name_formatting_fn():
    return StringMetricName


def get_local_training_statistics(
    model, user_dataset: AbstractDataset, config: GMMHyperParams,
    metric_name_current: Callable[[str], MetricName]
) -> Tuple[MappedVectorStatistics, Metrics]:
    assert isinstance(config, GMMHyperParams)

    likelihood, statistics = (model.get_mixture_statistics(
        user_dataset.raw_data, config.variance_scale,
        config.responsibility_scale))
    metrics = Metrics([(metric_name_current('log-likelihood before training'),
                        Weighted(likelihood.log_value,
                                 len(user_dataset.raw_data)))])

    return statistics, metrics


def test_global_gaussian(gmm_params, name_formatting_fn):
    num_dimensions = 2
    gaussian_1 = DiagonalGaussian([1, 2], [4, 5])
    gaussian_2 = DiagonalGaussian([2, 3], [7, 8])

    single_gaussian = GaussianMixtureModel(
        num_dimensions,
        Mixture([(1, diagonal_standard_gaussian(num_dimensions))]),
        cached_model_train_params=gmm_params)

    # Use integer counts so it is possible to correctly-distributed data to
    # check the global Gaussian.
    for mixture_parameters in [
        [(1, gaussian_1)],
        [(1, gaussian_1), (1, gaussian_2)],
        [(1, gaussian_1), (3, gaussian_2)],
    ]:
        # Generate data deterministically according to mixture_parameters.
        def component_statistics(parameters):
            for (count, gaussian) in parameters:
                dataset = unscented_user_data_set(num_dimensions,
                                                  gaussian.mean,
                                                  gaussian.variance)
                for _ in range(count):
                    statistics, _metrics = (get_local_training_statistics(
                        single_gaussian,
                        dataset,
                        gmm_params,
                        metric_name_current=name_formatting_fn))
                    yield statistics

        # Train a single Gaussian on this data.
        (retrained, _metrics) = single_gaussian.apply_model_update(
            any_sum(component_statistics(mixture_parameters)))
        ((_weight, expected_global_gaussian), ) = retrained.components

        # Actually call global_gaussian.
        mixture = GaussianMixtureModel(num_dimensions,
                                       Mixture(mixture_parameters))
        global_gaussian = mixture.global_gaussian()

        assert (global_gaussian.mean == pytest.approx(
            expected_global_gaussian.mean))
        assert (global_gaussian.variance == pytest.approx(
            expected_global_gaussian.variance))


def test_mix_up_simple():
    num_dimensions = 2

    # Initialise with a single unit Gaussian.
    model = GaussianMixtureModel(num_dimensions)
    assert len(model.components) == 1

    model_2 = model.mix_up(1)
    assert len(model_2.components) == 2
    model_3 = model_2.mix_up(2)
    assert len(model_3.components) == 4
    model_4 = model_3.mix_up(3)
    assert len(model_4.components) == 7


def test_training(gmm_params, name_formatting_fn):
    """
    Test get_local_training_statistics, apply_model_update, and mix_up.
    """
    num_dimensions = 2

    mean = np.asarray([3, 4])
    variance = np.asarray([4, 5])

    # Initialise with a single unit Gaussian.
    model = GaussianMixtureModel(num_dimensions,
                                 cached_model_train_params=gmm_params)

    # Train with data distributed according to (mean, variance).
    statistics, _metrics = get_local_training_statistics(
        model,
        unscented_user_data_set(2, mean, variance),
        config=gmm_params,
        metric_name_current=name_formatting_fn)

    model_2, _metrics_2 = model.apply_model_update(statistics)

    ((weight, gaussian), ) = model_2.components

    assert weight == 1
    assert gaussian.mean == pytest.approx(mean)
    assert gaussian.variance == pytest.approx(variance)

    # Make two components along the axis of greatest variance.
    model_3 = model_2.mix_up(num_extra_components=1)
    assert len(model_3.model.components) == 2

    # Train with data in two separate clusters.
    # These components are far enough that they should be reliably assigned to
    # either one or the other component.
    mean_left = np.asarray([2, -20])
    variance_left = np.asarray([2, 7])
    mean_right = np.asarray([4, 30])
    variance_right = np.asarray([4, 2])

    statistics_left, _metrics = get_local_training_statistics(
        model_3,
        unscented_user_data_set(2, mean_left, variance_left),
        gmm_params,
        metric_name_current=name_formatting_fn)
    statistics_right, _metrics = get_local_training_statistics(
        model_3,
        unscented_user_data_set(2, mean_right, variance_right),
        gmm_params,
        metric_name_current=name_formatting_fn)

    model_4, _metrics = model_3.apply_model_update(statistics_left +
                                                   statistics_right +
                                                   statistics_right)

    ((weight_left, component_left),
     (weight_right, component_right)) = model_4.model.components

    if component_right.mean[1] < component_left.mean[1]:
        weight_left, weight_right = weight_right, weight_left
        component_left, component_right = component_right, component_left

    assert weight_left == pytest.approx(1 / 3, rel=1e-4)
    assert weight_right == pytest.approx(2 / 3, rel=1e-4)

    assert component_left.mean == pytest.approx(mean_left, abs=0.01)
    assert component_left.variance == pytest.approx(variance_left, abs=0.1)

    assert component_right.mean == pytest.approx(mean_right, abs=0.01)
    assert component_right.variance == pytest.approx(variance_right, abs=0.1)
