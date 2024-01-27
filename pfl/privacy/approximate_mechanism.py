# Copyright © 2023-2024 Apple Inc.
"""
Approximate local privacy mechanisms with a central implementation for speed.
"""

import math
from abc import abstractmethod
from typing import Tuple

from pfl.context import CentralContext, UserContext
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics
from pfl.stats import TrainingStatistics

from .privacy_mechanism import CentralPrivacyMechanism, SplitPrivacyMechanism
from .privacy_snr import SNRMetric


class SquaredErrorLocalPrivacyMechanism(SplitPrivacyMechanism):
    """
    Abstract base class for a local privacy mechanism that knows its squared
    error.
    This can be used in two ways.

    First, one can use this information to analyse mechanisms.

    Second, the local mechanism can be approximated with a central mechanism
    with the same squared error.
    This central mechanism adds Gaussian noise to the sum of individual
    statistics, which is much faster in simulation. However, in live training,
    the local privacy mechanism should be applied on each device prior to
    sending the data back to the server for aggregation.
    Even if the distribution of the error on one individual contribution is not
    Gaussian, because of the central limit theorem on a reasonably-sized cohort
    the difference will usually not be noticeable.
    """

    @abstractmethod
    def sensitivity_scaling(self, num_dimensions: int) -> int:
        """
        Return scaling that needs to be applied to the output of
        ``constrain_sensitivity``.

        :param num_dimensions:
            The number of dimensions of the vector that this mechanism is
            applied on.
        """

    @abstractmethod
    def sensitivity_squared_error(self, num_dimensions: int,
                                  l2_norm: float) -> float:
        """
        Return the expected squared error that is caused by random behaviour of
        the ``constrain_sensitivity`` method.
        Note that this does not include error introduced by clipping.
        If ``add_noise`` scales the output of ``constrain_sensitivity``, that
        scaling does not have to be included.
        Instead just include it in ``sensitivity_scaling``.

        :param num_dimensions:
            The number of dimensions of the vector that this mechanism is
            applied on.
        :param l2_norm:
            The L2 norm of the vector that this mechanism is applied on.
        """

    @abstractmethod
    def add_noise_squared_error(self, num_dimensions: int,
                                cohort_size: int) -> float:
        """
        Return the expected squared error that is caused by the ``add_noise``
        method.

        :param num_dimensions:
            The number of dimensions of the vector that this mechanism is
            applied on.
        :param l2_norm:
            The L2 norm of the vector that this mechanism is applied on.
        """

    def get_squared_error(self, num_dimensions: int, l2_norm: float,
                          cohort_size: int) -> float:
        """
        Compute the expected squared error from applying this mechanism.

        :param num_dimensions:
            The number of dimensions of the vector that this mechanism is
            applied on.
        :param l2_norm:
            The L2 norm of the vector that this mechanism is applied on.
        :param cohort_size:
            The number of elements in the sum that this mechanism will be
            applied on.
            Set this to 1 for local privacy.
        """
        sensitivity_squared_error = self.sensitivity_squared_error(
            num_dimensions=num_dimensions, l2_norm=l2_norm)

        sensitivity_scaling = self.sensitivity_scaling(num_dimensions)

        add_noise_squared_error = self.add_noise_squared_error(
            num_dimensions=num_dimensions, cohort_size=cohort_size)
        return ((sensitivity_scaling**2 * sensitivity_squared_error) +
                add_noise_squared_error)

    def approximate_as_central_mechanism(self) -> CentralPrivacyMechanism:
        """
        Return an approximation of this mechanism that can be used as a central
        mechanism.
        To use this, imagine that ``local_privacy`` is the privacy mechanism to
        be approximated:

            central_privacy = local_privacy.approximate_as_central_mechanism()
            local_privacy = no_privacy

        ``central_privacy`` can then be passed into the backend as a central
        privacy mechanism, which can significantly speed up simulations when
        using local DP without affecting the outcomes of the simulations.

        :return:
            A central privacy mechanism that approximates the local privacy
            mechanism.
        """
        return GaussianApproximatedPrivacyMechanism(self)


class GaussianApproximatedPrivacyMechanism(CentralPrivacyMechanism):
    """
    Approximated version of a local privacy mechanism that can be applied as a
    central mechanism. This can make simulations much faster (but cannot be used
    in live training).

    To use this, imagine that ``local_mechanism`` is the privacy mechanism to be
    approximated and ``local_mechanism_config`` is its configuration::

        central_mechanism = local_mechanism.approximate_as_central_mechanism()
        central_mechanism_config = local_mechanism_config
        local_mechanism = NoPrivacy()

    ``central_mechanism`` can then be passed into the backend.

    :param local_mechanism:
        The local mechanism to be approximated.
    """

    def __init__(self, local_mechanism: SquaredErrorLocalPrivacyMechanism):
        super().__init__()
        self._local_mechanism = local_mechanism
        self._central_dp_clip_format_fn = self._central_metric_format(
            self._central_dp_clip_format_fn)
        self._central_dp_noise_format_fn = self._central_metric_format(
            self._central_dp_noise_format_fn)

    def _central_metric_format(self, name_formatting_fn):

        def format(n):  # noqa: A001
            return name_formatting_fn(f'(approx. local) {n}')

        return format

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        # Sensitivity is constrained for each user separately even when the
        # local privacy mechanism is approximated by a central one.
        return self._local_mechanism.constrain_sensitivity(
            statistics=stats,
            name_formatting_fn=self._central_dp_clip_format_fn,
            seed=user_context.seed)

    def postprocess_server(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:

        num_dimensions = stats.num_parameters
        sensitivity_scaling = self._local_mechanism.sensitivity_scaling(
            num_dimensions)
        # Scale up statistics.
        scaled_statistics = stats.apply_elementwise(
            lambda v: sensitivity_scaling * v)

        _metadata, weights = scaled_statistics.get_weights()
        signal_norm = get_ops().global_norm(weights, order=2)

        individual_squared_error = (
            self._local_mechanism.add_noise_squared_error(
                num_dimensions=num_dimensions, cohort_size=1))
        overall_squared_error = (central_context.cohort_size *
                                 individual_squared_error)

        noise_stddev = math.sqrt(overall_squared_error / num_dimensions)

        data_with_noise = scaled_statistics.apply(get_ops().add_gaussian_noise,
                                                  stddev=noise_stddev,
                                                  seed=central_context.seed)

        metrics = Metrics([
            (self._central_dp_noise_format_fn('signal-to-DP-noise ratio'),
             SNRMetric(signal_norm, overall_squared_error)),
            (self._central_dp_noise_format_fn('DP squared error'),
             overall_squared_error),
        ])

        return data_with_noise, metrics
