# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, Tuple, cast

from pfl.common_types import Population
from pfl.context import CentralContext, UserContext
from pfl.hyperparam.base import HyperParam, get_param_value
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import MetricName, Metrics, Weighted
from pfl.stats import MappedVectorStatistics, TrainingStatistics

from .gaussian_mechanism import GaussianMechanism
from .privacy_accountant import PrivacyAccountant
from .privacy_mechanism import CentrallyAppliedPrivacyMechanism


class MutableClippingBound(HyperParam[float]):
    """
    A mutable hyperparameter for clipping bound used in adaptive clipping.
    """

    def __init__(self, initial_clipping_bound: float):
        self._value = initial_clipping_bound

    def value(self) -> float:
        return self._value

    def set_value(self, value: float):
        self._value = value


class AdaptiveClippingGaussianMechanism(CentrallyAppliedPrivacyMechanism):
    """
    A `CentrallyAppliedPrivacyMechanism` class to implement adaptive clipping
    algorithm as in the paper: https://arxiv.org/pdf/1905.03871.pdf.

    The algorithm automatically adjust the clipping bound by optimizing
    P(‖x‖₂ ≤ C) = 𝛾, where ‖x‖₂ is the model update ℓ² norm, C is the clipping
    bound and 𝛾 is the `adaptive_clipping_norm_quantile`. For example, setting
    𝛾=0.1 and the algorithm will iteratively update the clipping bound C such
    that 10% of the device model update ℓ² norm will be less than C. Since
    the norms of model updates typically vary through the run (oftentimes
    decreasing over time), reducing the clipping and consequently also noise
    can be beneficial.

    The algorithm requires collecting clipping indicator (i.e. whether the model
    update is clipped or not) for estimating the quantile that clipping bound C
    tracks to optimize C to the desired quantile 𝛾. Clipping indicator is
    encoded as -1 or 1 on device and thus the estimated quantile equals averaged
    clipping indicators / 2 + 0.5. Central DP noise (standard deviation set by
    `clipping_indicator_noise_stddev`) is added to the aggregated clipping
    indicator to protect the privacy. Noisy aggregated clipping indicator will
    then be used to update clipping bound using geometric update rule with step
    size η, i.e. the `log_space_step_size` argument.

    :param make_gaussian_mechanism
        A function that makes a Gaussian Mechanism given a clipping bound as
        input. For example:

        :example:

            .. code-block:: python

                from pfl.privacy.gaussian_mechanism import GaussianMechanism
                make_gaussian_mechanism = lambda c: GaussianMechanism(c, 1.0)

    :param initial_clipping_bound:
        The initial ℓ² clipping bound for Gaussian Mechanism.
    :param clipping_indicator_noise_stddev:
        Standard deviation of Gaussian noise added to the aggregated clipping
        indicator. Recommended value is `0.1 * cohort_size` as suggested in
        https://arxiv.org/pdf/1905.03871.pdf.
    :param adaptive_clipping_norm_quantile:
        A quantile in [0, 1] representing the desired fraction of device model
        updates with ℓ² norm less than the clipping bound.
    :param log_space_step_size:
        Step size η for optimizing clipping bound in the log space. Clipping
        bound C is updated with logC ⟵ log(C exp(-ηg)) = logC - ηg where g is
        the derivative of quantile estimation loss. Recommended value for the
        step size η is 0.2 (default value) as suggested in
        https://arxiv.org/pdf/1905.03871.pdf.
    """  # noqa: RUF002

    # Hard-coded recipe key for clipping indicator scale matched with MLRuntime
    _CLIPPING_INDICATOR_SCALE_KEY = "ClippingIndicatorScale"
    # Name for clipping indicator
    _CLIPPING_INDICATOR_NAME = "adaptive_clipping/clipping_indicator"

    def __init__(self,
                 make_gaussian_mechanism: Callable[[MutableClippingBound],
                                                   GaussianMechanism],
                 initial_clipping_bound: float,
                 clipping_indicator_noise_stddev: float,
                 adaptive_clipping_norm_quantile: float,
                 log_space_step_size: float = 0.2):
        self._mutable_clipping_bound = MutableClippingBound(
            initial_clipping_bound)
        super().__init__(make_gaussian_mechanism(self._mutable_clipping_bound))
        assert isinstance(self._underlying_mechanism, GaussianMechanism)
        self._clipping_indicator_noise_stddev = clipping_indicator_noise_stddev
        self._adaptive_clipping_norm_quantile = adaptive_clipping_norm_quantile
        self._log_space_step_size = log_space_step_size

    @property
    def mutable_clipping_bound(self) -> MutableClippingBound:
        return self._mutable_clipping_bound

    def _get_overall_clipping_bound(self) -> float:
        # Increase clipping bound to accommodate the additional bit that is
        # used to adapt the clipping bound. Notation:
        #     C_delta: clipping bound for model updates (deltas), which is
        #         the parameter self._mutable_clipping_bound
        #     C_b: scale of the clipping indicator
        #     C: overall clipping bound
        #     m: cohort size
        #     sigma_c: overall sigma for noise for the new bit
        #     sigma_b: per-device sigma for noise for the extra bit
        #     sigma_g: cohort sigma for model updates
        #     r: r = sigma_g / sigma_c
        # Then, C = sqrt(C_delta^2 + C_b^2) = C_delta / sqrt(1-r^2)

        delta_noise_stddev = self._underlying_mechanism.relative_noise_stddev
        r = delta_noise_stddev / self._clipping_indicator_noise_stddev
        return get_param_value(
            self._mutable_clipping_bound) / math.sqrt(1 - r**2)

    def _get_clipping_indicator_scale(self) -> float:
        # Calculate the scale of the clipping indicator.
        # Using the same notation as in ``_get_overall_clipping_bound``, this
        # function computes
        #     C_b = C * sigma_g / sigma_c = C_delta * r / sqrt(1-r^2)
        delta_noise_stddev = self._underlying_mechanism.relative_noise_stddev
        r = delta_noise_stddev / self._clipping_indicator_noise_stddev
        assert r < 1.0, (
            "Model update stddev: {} >= clipping indicator stddev: {}, which "
            "results in invalid clipping bound. Please increase the value of "
            "norm_quantile_noise_stddev.".format(
                self._underlying_mechanism.relative_noise_stddev,
                self._clipping_indicator_noise_stddev))
        return r / math.sqrt(1 - r**2) * get_param_value(
            self._mutable_clipping_bound)

    def _geometric_update(self, noisy_norm_quantile: float):
        """
        Update clipping bound in log space.
        C ← C · exp(-ηC (γ_emp - γ)) where γ_emp is the (noisy) empirical
        fraction of devices that had model updates <= C.
        """  # noqa: RUF002
        updated_clipping_bound = get_param_value(
            self._mutable_clipping_bound) * math.exp(
                -self._log_space_step_size *
                (noisy_norm_quantile - self._adaptive_clipping_norm_quantile))
        self._mutable_clipping_bound.set_value(updated_clipping_bound)

    def postprocess_one_user(
            self, *, stats: TrainingStatistics,
            user_context: UserContext) -> Tuple[TrainingStatistics, Metrics]:
        """
        Postprocess the user local statistics by appending clipping indicator.
        """
        stats, metrics = self._underlying_mechanism.constrain_sensitivity(
            statistics=stats,
            name_formatting_fn=self._central_dp_clip_format_fn,
            seed=user_context.seed)
        stats = cast(MappedVectorStatistics, stats)
        clipped = cast(
            Weighted, metrics[self._central_dp_clip_format_fn(
                'fraction of clipped norms')]).overall_value
        # Match with on-device encoding
        clipping_indicator = 2 * (1 - clipped) - 1
        assert clipping_indicator in {-1.0, 1.0}
        stats[self._CLIPPING_INDICATOR_NAME] = get_ops().to_tensor(
            [clipping_indicator * self._get_clipping_indicator_scale()])
        return stats, metrics

    def _get_noisy_norm_quantile(self, stats: MappedVectorStatistics) -> float:
        """
        Popping the clipping indicator from `stats`; adding noise to the
        aggregated clipping indicator; scaling clipping indicator back to
        quantile.
        """
        assert self._CLIPPING_INDICATOR_NAME in stats
        # Pop clipping indicator from aggregated statistics, and average
        data, weight = stats.pop(self._CLIPPING_INDICATOR_NAME)
        clipping_indicator = get_ops().to_numpy(data / weight).item()
        # Scale clipping indicator to norm quantile, i.e. the noised fraction
        # of devices with model update norm less than clipping bound.
        # `noisy_norm_quantile` is b̃ from https://arxiv.org/pdf/1905.03871.pdf
        noisy_norm_quantile = float(
            clipping_indicator / self._get_clipping_indicator_scale() + 1) / 2
        return noisy_norm_quantile

    def postprocess_server(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:
        """
        Postprocess the aggregated statistics by adding Gaussian noise, popping
        clipping indicator and updating central clipping bound.

        :param stats:
            Aggregated model updates with clipping indicator appended.
        :param central_context:
            CentralContext with clipping bound to be updated.
        :return:
            A tuple `(popped_statistics, metrics)`:
            `popped_statistics` is the input model update with clipping
            indicator popped.
            `metrics` is a dict `description: value` where `value`
            contains the noisy norm quantile aggregated from devices and the
            updated norm bound.
        """
        prev_value = get_param_value(self._mutable_clipping_bound)
        self._mutable_clipping_bound.set_value(
            self._get_overall_clipping_bound())
        stats, mechanism_metrics = self._underlying_mechanism.add_noise(
            statistics=stats,
            cohort_size=central_context.cohort_size,
            name_formatting_fn=self._central_dp_noise_format_fn,
            seed=central_context.seed)
        self._mutable_clipping_bound.set_value(prev_value)
        stats = cast(MappedVectorStatistics, stats)
        noisy_norm_quantile = self._get_noisy_norm_quantile(stats)
        self._geometric_update(noisy_norm_quantile)
        return stats, mechanism_metrics | Metrics([(MetricName(
            "norm quantile",
            Population.TRAIN), Weighted.from_unweighted(noisy_norm_quantile))])

    def postprocess_server_live(
            self, *, stats: TrainingStatistics,
            central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[TrainingStatistics, Metrics]:
        """
        Postprocess the aggregated statistics by popping clipping indicator and
        updating central clipping bound.

        :param stats:
            Aggregated model updates with clipping indicator appended.
        :param central_context:
            CentralContext with clipping bound to be updated.
        :return:
            A tuple `(popped_statistics, metrics)`:
            `popped_statistics` is the input model update with clipping
            indicator popped.
            `metrics` is a dict `description: value` where `value`
            contain the noisy norm quantile aggregated from devices and the
            updated norm bound.
        """
        stats = cast(MappedVectorStatistics, stats)
        noisy_norm_quantile = self._get_noisy_norm_quantile(stats)
        self._geometric_update(noisy_norm_quantile)
        return stats, Metrics([(MetricName("norm quantile", Population.TRAIN),
                                Weighted.from_unweighted(noisy_norm_quantile))
                               ])

    @classmethod
    def construct_from_privacy_accountant(
        cls,
        accountant: PrivacyAccountant,
        initial_clipping_bound: float,
        clipping_indicator_noise_stddev: float,
        adaptive_clipping_norm_quantile: float,
        log_space_step_size: float = 0.2
    ) -> 'AdaptiveClippingGaussianMechanism':
        """
        Construct an instance of `AdaptiveClippingGaussianMechanism` from a
        privacy accountant.
        """
        # TODO is this correct?
        assert accountant.mechanism == 'gaussian', (
            'Only Gaussian mechanism is supported with Adaptive Clipping')
        make_gaussian_mechanism = lambda clipping_bound: GaussianMechanism.from_privacy_accountant(  # pylint: disable=line-too-long
            accountant=accountant,
            clipping_bound=clipping_bound)
        return cls(
            make_gaussian_mechanism=make_gaussian_mechanism,
            initial_clipping_bound=initial_clipping_bound,
            clipping_indicator_noise_stddev=clipping_indicator_noise_stddev,
            adaptive_clipping_norm_quantile=adaptive_clipping_norm_quantile,
            log_space_step_size=log_space_step_size)
