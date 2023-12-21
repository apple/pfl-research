# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import itertools
import os
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, List, Optional, Tuple

import numpy as np
from pfl.data.dataset import AbstractDatasetType

from pfl.hyperparam import ModelHyperParams
from pfl.internal.distribution import DiagonalGaussian, LogFloat, Mixture
from pfl.internal.distribution.distribution import (Distribution, any_product,
                                                    any_sum)
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.model.base import EvaluatableModel
from pfl.stats import MappedVectorStatistics


@dataclass(frozen=True)
class GMMHyperParams(ModelHyperParams):
    """
    Configuration for training Gaussian mixture models.

    :param variance_scale:
        Scaling factor for the part of the statistics vector where the variance
        statistics go.
        This is relative to the mean, which is not scaled.
        Scaling parts of the statistics can effectively reduce noise for
        differential privacy.
    :param responsibility_scale:
        Scaling factor for the part of the statistics vector where the
        responsibilities go.
        This is relative to the mean, which is not scaled.
        Scaling parts of the statistics can effectively reduce noise for
        differential privacy.
    :param variance_floor_fraction:
        Floor to the variance of each component when training, as a fraction of
        the global variance.
    :param minimum_component_weight:
        Weight below which components are pruned when training.
    """
    variance_scale: float = 1.
    responsibility_scale: float = 1.
    variance_floor_fraction: float = 0.01
    minimum_component_weight: float = 0.01


class GaussianMixtureModel(EvaluatableModel):
    """
    A density model represented by a mixture of diagonal-covariance Gaussians.
    This model is immutable: ``apply_model_update`` only returns a new model,
    and it does not change the object.

    Training uses standard expectation-maximization, which approximates
    maximum-likelihood training.
    The statistics are the standard statistics, but transformed so that they
    are efficient when noise is added for differential privacy.

    :param num_dimension:
        The length of the vectors that this model is over.
    :param model:
        The underlying ``Mixture[DiagonalGaussian]``.
    :param cached_model_train_params:
        Used internally.
    """

    # pytype: disable=missing-parameter
    def __init__(self,
                 num_dimensions: int,
                 model: Optional[Mixture] = None,
                 *,
                 cached_model_train_params: Optional[GMMHyperParams] = None):
        # pytype: enable=missing-parameter
        """
        Initialise a mixture of Gaussians.
        If ``model is None``, the mixture has one component: the unit Gaussian.
        """
        self._num_dimensions = num_dimensions

        if model is None:
            mean = np.zeros(shape=(num_dimensions, ))
            variance = np.ones(shape=(num_dimensions, ))

            self._model = Mixture([(1., DiagonalGaussian(mean, variance))])
        else:
            self._model = model

        self._cached_model_train_params = cached_model_train_params

    @property
    def allows_distributed_evaluation(self) -> Optional[bool]:
        # No metrics require postprocessing by each user.
        return True

    def global_gaussian(self) -> DiagonalGaussian:
        """
        :return:
            The global Gaussian, as computed from the mixture.
            Note that this is exact, in the sense that if the data that was
            used to train the mixture was used to train a single Gaussian,
            the global Gaussian would be that single Gaussian.
            (Here, "train" means "find the parameters that maximize the
            likelihood".)
        """
        global_mean = any_sum(weight * g.mean
                              for (weight, g) in self._model.components)
        global_variance = any_sum(
            weight * (g.variance + np.square(g.mean - global_mean))
            for (weight, g) in self._model.components)

        return DiagonalGaussian(global_mean, global_variance)

    @property
    def model(self) -> Mixture:  # pytype: disable=missing-parameter  # pylint: disable=line-too-long
        """
        :return:
            The underlying ``Mixture`` of ``DiagonalGaussian`` s.
        """
        return self._model

    @property
    def components(self) -> List[Tuple[float, Distribution]]:
        """
        :return:
            The underlying components, as a list of tuples with weights and
            components.
            The weights add up to 1.
        """
        return self._model.components

    def _statistics_normalization(self, old_gaussian: DiagonalGaussian):
        """
        Compute the expected values and approximate ranges of the mean and
        variance statistics.
        These are used to normalize the statistics.
        :return:
            A tuple ``((expected_mean, mean_range), (expected_variance,
            variance_range))``
        """
        return ((old_gaussian.mean, np.sqrt(old_gaussian.variance)),
                (np.square(old_gaussian.mean) + old_gaussian.variance,
                 old_gaussian.variance))

    def _gaussian_statistics_single_point(
            self, old_gaussian: DiagonalGaussian,
            point: np.ndarray) -> Tuple[LogFloat, MappedVectorStatistics]:
        """
        :return:
            The statistics to train the Gaussian for a single point.
            These are the first- and second-order statistics, but both with
            affine transformations applied to them, for efficiency under
            differential privacy.

            The mean statistics are given relative to the old mean, and divided
            by the old standard deviation.

            The variance statistics are given relative to what they should be
            from the old mean and variance, and divided by the old variance.
        """
        point = np.asarray(point)
        assert len(point.shape) == 1
        likelihood = old_gaussian.density(point)

        ((expected_mean, mean_range),
         (expected_variance,
          variance_range)) = self._statistics_normalization(old_gaussian)

        relative_mean_statistic = ((point - expected_mean) / mean_range)

        relative_variance_statistic = ((np.square(point) - expected_variance) /
                                       variance_range)

        statistics = MappedVectorStatistics(
            {
                'mean': relative_mean_statistic,
                'variance': relative_variance_statistic
            },
            weight=1.)
        return likelihood, statistics

    def _mixture_statistics_single_point(
        self, point: np.ndarray, variance_scale: float,
        responsibility_scale: float
    ) -> Tuple[LogFloat, MappedVectorStatistics]:
        """
        :return:
            The statistics to train the mixture of Gaussians for a single point.
        """

        def posteriors_statistics(
        ) -> Generator[Tuple[LogFloat, MappedVectorStatistics], None, None]:
            for weight, component in self.model.components:
                component_likelihood, statistics = (
                    self._gaussian_statistics_single_point(component, point))
                yield (LogFloat.from_value(weight) * component_likelihood,
                       statistics)

        unnormalized_posteriors: List[LogFloat]
        unweighted_statistics: List[MappedVectorStatistics]
        unnormalized_posteriors, unweighted_statistics = zip(
            *posteriors_statistics())

        # In EM training of mixture distributions, the "responsibilities" are
        # the posteriors (normalized), and they are used to weight the
        # per-component statistics.
        likelihood: LogFloat = any_sum(unnormalized_posteriors)
        responsibilities = [(posterior / likelihood).value
                            for posterior in unnormalized_posteriors]

        weighted_statistics = {}
        for (component_index,
             (responsibility, component_statistics)) in enumerate(
                 zip(responsibilities, unweighted_statistics)):
            weighted_statistics[f'responsibility-{component_index}'] = (
                np.asarray([responsibility_scale * responsibility]))
            weighted_statistics[f'mean-{component_index}'] = (
                responsibility * component_statistics['mean'])
            weighted_statistics[f'variance-{component_index}'] = (
                variance_scale * responsibility *
                component_statistics['variance'])
        return likelihood, MappedVectorStatistics(weighted_statistics,
                                                  weight=1)

    def get_mixture_statistics(self, points: Iterable[np.ndarray],
                               variance_scale: float,
                               responsibility_scale: float):
        """
        :return:
            The statistics to train the mixture of Gaussians for multiple
            points.
        """
        likelihoods, statistics = zip(*(self._mixture_statistics_single_point(
            point, variance_scale, responsibility_scale) for point in points))
        return (any_product(likelihoods), any_sum(statistics))

    def _maximum_likelihood_gaussians(
        self, statistics: MappedVectorStatistics, variance_scale: float,
        responsibility_scale: float, variance_floor: float
    ) -> Generator[Tuple[float, DiagonalGaussian], None, None]:
        """
        Deserialise statistics as produced by _serialise_statistics and train
        with them.
        """

        overall_weight = statistics.weight

        for component_index, (_, old_gaussian) in enumerate(self.components):
            responsibility_name = f'responsibility-{component_index}'
            mean_name = f'mean-{component_index}'
            variance_name = f'variance-{component_index}'

            responsibility = (float(statistics[responsibility_name]) /
                              responsibility_scale)
            weight = (responsibility / overall_weight)

            ((expected_mean, mean_range),
             (expected_variance,
              variance_range)) = self._statistics_normalization(old_gaussian)

            # Undo the transformations from _gaussian_statistics_single_point.
            mean = (statistics[mean_name] / responsibility * mean_range +
                    expected_mean)

            variance = (statistics[variance_name] / variance_scale /
                        responsibility * variance_range + expected_variance -
                        np.square(mean))

            floored_variance = np.maximum(variance_floor, variance)

            yield weight, DiagonalGaussian(mean, floored_variance)

    def apply_model_update(
        self, statistics: MappedVectorStatistics
    ) -> Tuple['GaussianMixtureModel', Metrics]:
        assert self._cached_model_train_params.variance_scale
        variance_floor = (
            self._cached_model_train_params.variance_floor_fraction *
            self.global_gaussian().variance)
        weights_gaussians = list(
            self._maximum_likelihood_gaussians(
                statistics,
                variance_scale=self._cached_model_train_params.variance_scale,
                responsibility_scale=self._cached_model_train_params.
                responsibility_scale,
                variance_floor=variance_floor))

        minimum_weight = (
            self._cached_model_train_params.minimum_component_weight)
        pruned_weights_gaussians = [(weight, gaussian)
                                    for (weight, gaussian) in weights_gaussians
                                    if weight >= minimum_weight]

        new_gmm = GaussianMixtureModel(
            self._num_dimensions,
            Mixture(pruned_weights_gaussians),
            cached_model_train_params=self._cached_model_train_params)

        return (new_gmm, Metrics())

    def evaluate(self,
                 dataset: AbstractDatasetType,
                 name_formatting_fn: Callable[[str], StringMetricName],
                 eval_params: Optional[ModelHyperParams] = None) -> Metrics:
        overall_likelihood = LogFloat.from_value(1)
        for datapoint in dataset.raw_data:
            likelihood = self._model.density(datapoint)
            overall_likelihood = overall_likelihood * likelihood

        metric = Weighted(overall_likelihood.log_value, len(dataset.raw_data))
        return Metrics([(name_formatting_fn('log-likelihood'), metric)])

    def mix_up(self, num_extra_components: int) -> 'GaussianMixtureModel':
        """
        Introduce extra components in a heuristic manner.

        The new components are generated by splitting up the heaviest
        components.

        :param num_extra_components:
            The number of components to add, i.e. the number of components to
            split.
            If this is greater than the current number of components, then all
            components are split once, and the number of components is merely
            doubled.

        :return:
            The new model with extra components.
        """

        def partition():
            """
            :return:
                A tuple with a list of the heaviest components and a list of
                the least heavy ones.
            """
            real_num_extra_components = min(len(self.model.components),
                                            num_extra_components)
            # Sort by component weight.
            sorted_components = sorted(
                self.model.components,
                key=lambda weight_component: weight_component[0],
                reverse=True)
            return (sorted_components[:real_num_extra_components],
                    sorted_components[real_num_extra_components:])

        components_to_duplicate, static_components = partition()

        def duplicate_components(components: List[Tuple[float, Distribution]]):
            for weight, component in components:
                component_1, component_2 = component.split(offset=.5)
                yield .5 * weight, component_1
                yield .5 * weight, component_2

        new_mixture = Mixture(
            itertools.chain(duplicate_components(components_to_duplicate),
                            static_components))
        return GaussianMixtureModel(
            self._num_dimensions,
            new_mixture,
            cached_model_train_params=self._cached_model_train_params)

    def save(self, dir_path: str) -> None:
        import pickle
        output_file_name = os.path.join(dir_path, 'trained_gmm.pickle')
        with open(output_file_name, 'wb') as output_file:
            pickle.dump(self._model, output_file)
