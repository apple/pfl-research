# Copyright © 2023-2024 Apple Inc.
"""
Apply differential privacy to statistics.
"""

# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pfl.context import CentralContext, UserContext
from pfl.hyperparam import HyperParamClsOrFloat, get_param_value
from pfl.internal.ops.numpy_ops import NumpySeedScope
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.postprocessor.base import Postprocessor
from pfl.stats import TrainingStatistics


@dataclass(frozen=True, order=True)
class PrivacyMetricName(StringMetricName):
    """
    A structured name for privacy metrics which includes whether it was
    generated using a local privacy mechanism or central privacy mechanism.

    :param description:
        The metric name represented as a string.
    :param is_local_privacy:
        `True` if metric is related to local DP, `False` means central DP.
    :param on_summed_stats:
        `True` if metric is calculated on summed stats.
        Usually only true for the noise operation of central DP.
    """
    is_local_privacy: bool
    on_summed_stats: bool = False

    def __str__(self) -> str:
        prefix = 'Local DP | ' if self.is_local_privacy else 'Central DP | '
        postfix = '' if not self.on_summed_stats else ' on summed stats'
        return f'{prefix}{self.description}{postfix}'


class PrivacyMechanism(ABC, Postprocessor):
    """
    Base class for privacy mechanisms.
    """
    pass


class LocalPrivacyMechanism(PrivacyMechanism):
    """
    Base class for mechanisms that convert statistics into their
    local differentially private version.
    This will often perform clipping and then add noise.

    Bounds on the privacy loss (for example, epsilon and delta) should be passed
    in as parameters when constructing the object.
    These are the parameters that would be baked in on device.
    """

    @abstractmethod
    def privatize(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Take unbounded statistics from one individual and turn them directly
        into statistics that protect the individual's privacy using
        differential privacy.

        :param statistics:
            The statistics to be made differentially private.
        :param name_formatting_fn:
            Function that formats a metric name appropriately.
        :param seed:
            An int representing a seed to use for random noise operations.
            This is useful to avoid generating the same noise if there are
            replicated workers.
        :return:
            A tuple `(noised_statistics, metrics)`
            `noised_statistics` is a new
            :class:`~pfl.stats.TrainingStatistics`, that is a version of
            `statistics` that has been constrained and has had
            noise added to it (i.e., it has been privatized).
            `metrics` is a `Metrics` object with `name: value` where `value`
            can be useful to display or analyse.
            For example, this could have statistics on the properties of the
            noise added.
        """

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        name_format_fn = lambda n: PrivacyMetricName(n, is_local_privacy=True)
        # pytype: enable=duplicate-keyword-argument,wrong-arg-count
        # Privatize statistics as local postprocessing.
        return self.privatize(stats, name_format_fn, user_context.seed)

    def postprocess_server(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:
        # Nothing to do on the server for local DP mechanisms.
        return stats, Metrics()

    def postprocess_server_live(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:
        # Nothing to do on the server for local DP mechanisms.
        return stats, Metrics()


class SplitPrivacyMechanism(LocalPrivacyMechanism):
    """
    Base class for privacy mechanism that works in two stages: (1) constrain
    the sensitivity; (2) add noise.

    This is the case for many mechanisms.
    Some of those can be used for central privacy, but not all.
    Even where not, they can sometimes be approximated by a central privacy
    mechanism.
    """

    @abstractmethod
    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Constrain the sensitivity of the statistics, e.g. by norm-clipping.
        This makes it possible to determine the amount of noise necessary to
        guarantee differential privacy.

        :param statistics:
            The statistics that need to be constrained.
        :param name_formatting_fn:
            Function that formats a metric name appropriately.
        :param seed:
            An int representing a seed to use for random noise operations.
            This is useful to avoid generating the same noise if there are
            replicated workers.
        :return:
            A tuple `(constrained_statistics, metrics)`.
            `constrained_statistics` is a new ``TrainingStatistics``
            which is a version `statistics` that adheres to the sensitivity.
            `metrics` is a dict `description: value` where `value` is a value
            that can be useful to display or analyse.
            For example, this could have statistics on the clipping performed.
        """

    @abstractmethod
    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Transform statistics to protect the privacy of the data with
        differential privacy, for example by adding noise.
        It is assumed that the contribution of any individual user to
        `statistics` has been limited using `constrain_sensitivity`.

        :param statistics:
            The statistics to be made differentially private.
        :param cohort_size:
            The number of individuals whose data has gone into `statistics`.
            This is required in particular for approximations of local DP.
        :param name_formatting_fn:
            Function that formats a metric name appropriately.
        :param seed:
            An int representing a seed to use for random noise operations.
            This is useful to avoid generating the same noise if there are
            replicated workers.

        :return:
            A tuple `(noised_statistics, metrics)`
            `noised_statistics` is a new
            :class:`~pfl.stats.TrainingStatistics`, that
            is a clipped/noised version of `clipped_statistics`.
            `metrics` is a `Metrics` object with `name: value` where `value`
            can be useful to display or analyse.
            For example, this could have statistics on the properties of the
            noise added.
        """

    def privatize(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        # Implement local DP in terms of the above.

        with NumpySeedScope(seed):
            sensitivity_seed = np.random.randint(0, 2**32 - 1)
            noise_seed = np.random.randint(0, 2**32 - 1)

        (constrained_statistics,
         sensitivity_metrics) = self.constrain_sensitivity(
             statistics=statistics,
             name_formatting_fn=name_formatting_fn,
             seed=sensitivity_seed)

        noised_statistics, noise_metrics = self.add_noise(
            statistics=constrained_statistics,
            cohort_size=1,
            name_formatting_fn=name_formatting_fn,
            seed=noise_seed)

        metrics = sensitivity_metrics | noise_metrics
        return noised_statistics, metrics


class CentrallyApplicablePrivacyMechanism(SplitPrivacyMechanism):
    """
    Base class for local privacy mechanisms that can be applied centrally
    to approximate the local privacy mechanism more efficiently. Classes
    representing such mechanisms should derive from this.

    To apply the mechanism centrally, ``constrain_sensitivity`` will be called
    on each contribution, and ``add_noise`` on the aggregate.
    """
    pass


class CentralPrivacyMechanism(PrivacyMechanism):
    """
    Base class for differential privacy mechanisms which provides central
    differential privacy.

    This means that ``postprocess_one_user`` may apply processing to ensure
    sensitivity, and ``postprocess_server`` will transform the aggregate
    statistics randomly.
    """

    def __init__(self):
        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        self._central_dp_clip_format_fn = lambda n: PrivacyMetricName(
            n, is_local_privacy=False)
        self._central_dp_noise_format_fn = lambda n: PrivacyMetricName(
            n, is_local_privacy=False, on_summed_stats=True)
        # pytype: enable=duplicate-keyword-argument,wrong-arg-count


class CentrallyAppliedPrivacyMechanism(CentralPrivacyMechanism):
    """
    Wrap a local privacy mechanism (which is a local privacy mechanism by
    default), and transform it into a central privacy mechanism. The wrapped
    mechanism is transformed to perform ``constrain_sensitivity`` to
    individual contributions and ``add_noise`` on the aggregated statistics
    server-side.

    This also means that in the standard case, scaling can happen in
    ``constrain_sensitivity`` and be undone in ``add_noise``.
    For example, ``constrain_sensitivity`` can choose to clip to a clipping
    bound, or to 1.
    In the latter case, ``add_noise`` should probably scale by the clipping
    bound.

    :param underlying_mechanism:
        The privacy mechanism (which is a local privacy mechanism by default)
        to transform into a central privacy mechanism.
    """

    def __init__(self,
                 underlying_mechanism: CentrallyApplicablePrivacyMechanism):
        super().__init__()
        self._underlying_mechanism = underlying_mechanism

        assert isinstance(underlying_mechanism,
                          CentrallyApplicablePrivacyMechanism), (
                              'The mechanism to wrap must derive from '
                              'CentrallyApplicablePrivacyMechanism')

    @property
    def underlying_mechanism(self):
        return self._underlying_mechanism

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:

        return self._underlying_mechanism.constrain_sensitivity(
            statistics=stats,
            name_formatting_fn=self._central_dp_clip_format_fn,
            seed=user_context.seed)

    def postprocess_server(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:

        return self._underlying_mechanism.add_noise(
            statistics=stats,
            cohort_size=central_context.cohort_size,
            name_formatting_fn=self._central_dp_noise_format_fn,
            seed=central_context.seed)

    def postprocess_server_live(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:
        return stats, Metrics()


class NoPrivacy(CentrallyApplicablePrivacyMechanism):
    """
    Dummy privacy mechanism that does not do anything, but presents the same
    interface as real privacy mechanisms. This is useful for testing
    functionality of the code without the impact of a privacy mechanism.
    """

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        :return:
            A tuple `(statistics, metrics)`.
            `statistics` is the input unchanged.
            `metrics` is empty.
        """
        return statistics, Metrics()

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        :return:
            A tuple `(statistics, metrics)`.
            `statistics` is the input unchanged.
            `metrics` is empty.
        """
        return statistics, Metrics()


class NormClipping(SplitPrivacyMechanism):
    """
    Constrain the sensitivity of an individual's data by clipping the ℓp norm.
    This clipping is the first step in many privacy mechanisms.
    This class implements one half of
    :class:`~pfl.privacy.privacy_mechanism.LocalPrivacyMechanism`.

    :param order:
        The order of the norm.
        This must be a positive integer (e.g., `1` or `2`) or np.inf.
    :param clipping_bound:
        The norm bound for clipping.
    """

    def __init__(self, order: float, clipping_bound: HyperParamClsOrFloat):
        self._order = order
        self._clipping_bound = clipping_bound
        assert order > 0.0
        assert (order % 1 == 0 or order
                == float('inf')), "Order must be a natural number or infinity"

    @property
    def clipping_bound(self):
        return self._clipping_bound

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        :return:
            Statistics with their overall norm bounded by ``clipping_bound``.
            The norm of these statistics may be less.
        """
        clipping_bound = get_param_value(self._clipping_bound)

        _metadata, weights = statistics.get_weights()
        global_norm = get_ops().global_norm(weights, order=self._order)

        # Compute the number of times clipping occurs (out of 1).
        clipped_count = int(global_norm > clipping_bound)

        # Normalise all arrays at once.
        # Otherwise the noise would have to be computed separately for arrays,
        # and then the epsilons add up.
        if clipped_count:
            statistics = statistics.apply_elementwise(
                lambda v: v * (clipping_bound / global_norm))

        metrics = Metrics([
            (name_formatting_fn(f'l{self._order:.0f} norm bound'),
             Weighted.from_unweighted(clipping_bound)),
            (name_formatting_fn('fraction of clipped norms'),
             Weighted.from_unweighted(clipped_count)),
            (name_formatting_fn('norm before clipping'),
             Weighted.from_unweighted(global_norm)),
        ])
        return (statistics, metrics)


class NormClippingOnly(NormClipping, NoPrivacy):
    """
    Dummy privacy mechanism that does not do any privacy but only lp-norm
    clipping, but presents the same interface as real privacy mechanisms.
    This is useful for testing the impact of clipping only.

    :param order:
        The order of the norm.
        This must be a positive integer (e.g., `1` or `2`) or np.inf.
    :param clipping_bound:
        The norm bound for clipping.
    """

    def __init__(self, order: float, clipping_bound: HyperParamClsOrFloat):
        NormClipping.__init__(self, order, clipping_bound)
        NoPrivacy.__init__(self)
