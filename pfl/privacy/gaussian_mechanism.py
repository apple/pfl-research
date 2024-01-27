# Copyright © 2023-2024 Apple Inc.
'''
The Gaussian mechanism for differential privacy.
'''

from typing import Optional, Tuple

from pfl.hyperparam import HyperParamClsOrFloat, get_param_value
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from . import compute_parameters
from .approximate_mechanism import SquaredErrorLocalPrivacyMechanism
from .privacy_accountant import PrivacyAccountant
from .privacy_mechanism import CentrallyApplicablePrivacyMechanism, NormClipping
from .privacy_snr import SNRMetric


def get_noise_stddev(clipping_bound: HyperParamClsOrFloat,
                     relative_noise_stddev: float) -> float:
    return get_param_value(clipping_bound) * relative_noise_stddev


class GaussianMechanism(CentrallyApplicablePrivacyMechanism,
                        SquaredErrorLocalPrivacyMechanism, NormClipping):
    """
    Apply the Gaussian mechanism for differential privacy, which consists
    of scaling the statistics down to make their ℓ² norm smaller or equal
    than the clipping bound parameter, and adding Gaussian noise.

    The ℓ² norm is computed over all the arrays in `statistics` (as if these
    were concatenated into a single vector). If the norm is greater than the
    clipping bound, the statistics are scaled down linearly so that the
    resulting ℓ² norm is equal to the bound.
    If the ℓ² norm is below the bound, the original values are passed through
    unaltered.

    The Gaussian noise is then added to each statistic with the scale defined
    by the relative_noise_stddev as described below.

    To ensure (epsilon, delta) differential privacy, it is important to ensure
    that relative_noise_stddev is sufficiently large.  To initialize this class
    using `epsilon` and `delta`, rather than using the standard deviation of the
    Gaussian noise, use the method `construct_single_iteration` (this sets the
    noise standard deviation automatically to ensure (epsilon, delta) DP).

    The differential privacy guarantee assumes that each user participates
    in the training at most once. For multiple iterations (which is the
    typical case in Federated Learning), it is recommended to use
    ``from_privacy_accountant``.

    :param clipping_bound:
        The ℓ² norm bound for clipping the statistics (e.g. model updates) using
        ``constrain_sensitivity`` before sending them back to the server.
    :param relative_noise_stddev:
        The standard deviation of the Gaussian noise added to each statistic is
        defined as ``relative_noise_stddev * clipping_bound``. The standard deviation
        thus increases linearly with the clipping bound and the multiplier is
        given by this parameter ``relative_noise_stddev``.
    """

    def __init__(self, clipping_bound: HyperParamClsOrFloat,
                 relative_noise_stddev: float):
        NormClipping.__init__(self, 2., clipping_bound)
        self._relative_noise_stddev = relative_noise_stddev
        self._privacy_accountant: Optional[PrivacyAccountant] = None

    @property
    def relative_noise_stddev(self):
        return self._relative_noise_stddev

    @property
    def privacy_accountant(self) -> Optional[PrivacyAccountant]:
        return self._privacy_accountant

    def sensitivity_scaling(self, num_dimensions):
        return 1

    def sensitivity_squared_error(self, num_dimensions: int, l2_norm: float):
        # constrain_sensitivity does not introduce error due to randomness
        return 0.

    def add_noise_squared_error(self, num_dimensions: int, cohort_size: int):
        noise_stddev = get_noise_stddev(self._clipping_bound,
                                        self._relative_noise_stddev)
        return (noise_stddev**2) * num_dimensions

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        noise_stddev = get_noise_stddev(self._clipping_bound,
                                        self._relative_noise_stddev)
        data_with_noise = statistics.apply(get_ops().add_gaussian_noise,
                                           stddev=noise_stddev,
                                           seed=seed)

        num_dimensions = statistics.num_parameters
        _metadata, weights = statistics.get_weights()
        signal_norm = get_ops().global_norm(weights, order=2)
        squared_error = num_dimensions * (noise_stddev**2)

        metrics = Metrics([(name_formatting_fn('DP noise std. dev.'),
                            Weighted.from_unweighted(noise_stddev)),
                           (name_formatting_fn('signal-to-DP-noise ratio'),
                            SNRMetric(signal_norm, squared_error))])

        return data_with_noise, metrics

    @classmethod
    def construct_single_iteration(cls, clipping_bound: HyperParamClsOrFloat,
                                   epsilon: float,
                                   delta: float) -> 'GaussianMechanism':
        """
        Construct an instance of `GaussianMechanism` from an ε and a δ.
        This is suitable for giving out data once.
        If you apply the noise to the same individual's data multiple times, the
        privacy costs should be added up.

        :param clipping_bound:
            The norm bound for clipping.
        :param epsilon:
            The ε parameter of differential privacy.
            This gives an upper bound on the amount of privacy loss.
        :param delta:
            The δ (delta) parameter of (ε,δ)-differential privacy.
            This gives an upper bound on the probability that the privacy loss
            is more than ε.
        """
        # Calculate the required noise needed in order to guarantee
        # (epsilon,delta)-DP when clipping_bound = 1. The noise is later
        # multiplied by the actual clipping_bound.
        relative_noise_stddev = compute_parameters.AnalyticGM_robust(
            epsilon, delta, 1, 1.)

        return cls(clipping_bound, relative_noise_stddev)

    @classmethod
    def from_privacy_accountant(cls, accountant: PrivacyAccountant,
                                clipping_bound: HyperParamClsOrFloat):
        """
        Construct an instance of `GaussianMechanism` from an instance of
        `PrivacyAccountant`.
        """
        obj = cls(clipping_bound, accountant.cohort_noise_parameter)
        obj._privacy_accountant = accountant
        return obj
