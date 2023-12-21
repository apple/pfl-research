# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Classes that represent bounds on privacy loss, to represent differential
privacy.
"""

import math

import pfl.internal.logging_utils as logging_utils


class ApproximatePrivacyLossBound:
    """
    A bound on the privacy loss, in the sense of approximate differential
    privacy.
    This is parameterised with ε and δ.
    It is therefore often called (ε,δ) differential privacy.

    A lower ε and a lower δ indicates less loss of privacy.

    :param epsilon:
        An upper bound on the privacy loss.
    :param delta:
        The δ parameter of approximate differential privacy.
        Very loosely speaking, this is the probability that the upper bound of
        epsilon is exceeded.
    """

    def __init__(self, epsilon: float, delta: float):
        assert epsilon >= 0
        assert 0 <= delta < 1
        self._epsilon = epsilon
        self._delta = delta

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def delta(self):
        return self._delta

    def apply_multiple_times(self, step_num: int):
        """
        Convert this bound to one for losing privacy this multiple times.
        This yields only one of the possible guarantees, and it is often a loose
        bound.

        The guarantee that is returned has a higher epsilon and delta, since the
        privacy loss after multiple applications is greater.
        """
        return ApproximatePrivacyLossBound(step_num * self._epsilon,
                                           step_num * self._delta)

    def __lt__(self, other):
        """
        Return `True` if `self` guarantees less privacy loss than `other`.

        This implements involves a partial ordering: it is possible to have
        a lower epsilon and a higher delta, in which case the objects are
        incomparable.
        Then,  `self < other` and `other < self` both return `False`.
        """
        if self._epsilon < other._epsilon:
            return (self._delta <= other._delta)
        elif self._epsilon == other._epsilon:
            return (self._delta < other._delta)
        else:
            return False

    def __str__(self):
        return f'({self._epsilon:.5}, {self._delta:.5})-ADP'

    def __repr__(self):
        return (f'ApproximatePrivacyLossBound(epsilon={self._epsilon}, '
                f'delta={self._delta})')


class PrivacyLossBound:
    """
    A bound on the privacy loss, in the sense of pure differential privacy.
    This is parameterised with ε.

    A guarantee with a lower ε loses less privacy.

    :param epsilon:
        An upper bound on the privacy loss.
    """

    def __init__(self, epsilon: float):
        assert epsilon >= 0
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    def apply_multiple_times(self, step_num: int):
        """
        Convert this guarantee to one for applying this multiple times.
        This yields only one of the possible guarantees, and it is often a loose
        bound.

        The guarantee that is returned has a higher epsilon and delta, since the
        privacy loss after multiple iterations is greater.
        """
        return PrivacyLossBound(step_num * self._epsilon)

    def __lt__(self, other):
        """
        Return `True` if `self` guarantees less privacy loss than `other`.

        This implements involves a strict weak ordering.
        """
        return self._epsilon < other._epsilon

    def __str__(self):
        return f'{self._epsilon:.5}-DP'

    def __repr__(self):
        return (f'PrivacyLossBound(epsilon={self._epsilon})')


class RenyiPrivacyLossBound:
    """
    A bound for privacy loss in terms of Rényi differential privacy.
    This measures a Rényi divergence between the output of a mechanism applied
    on two adjacent databases.
    This is parameterised by a parameter α, the order of the Rényi divergence
    used, and the divergence ε, which is analogous but not the same as ε in
    standard or approximate DP.

    This is based on Mironov (2017), "Rényi Differential Privacy".
    https://arxiv.org/abs/1702.07476

    A bound for one order does not imply any bound for another order.
    However, it is possible to convert a bound for a specific order to one
    expressed as (ε,δ)-approximate differential privacy.

    :param order:
        The order of the Rényi divergence that is used.
        This must at least 1.
    :param epsilon:
        The bound on the value of the Rényi divergence that the mechanism
        provides.
    """

    def __init__(self, order: float, epsilon: float):
        self._order = order
        self._epsilon = epsilon
        assert order >= 1
        assert epsilon >= 0

    @property
    def order(self) -> float:
        return self._order

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def apply_multiple_times(self, num_applications: int):
        """
        Convert this bound to one for applying this multiple times.

        The bound that is returned has a higher epsilon, since the privacy
        loss after multiple applications is greater.
        """
        return RenyiPrivacyLossBound(self.order,
                                     num_applications * self.epsilon)

    def convert_to_approximate_dp(
            self, desired_delta: float) -> ApproximatePrivacyLossBound:
        """
        Convert a bound in terms of the (α,ε)-Rényi differential privacy
        into a bound in terms of (ε,δ)-approximate differential privacy.

        The desired δ parameter is given, and the corresponding ε is returned.

        This implements Mironov (2017), Proposition 3.
        It may well be possible to improve the bound that this relies on in the
        future.

        :param order:
            The order of the Rényi differential privacy.
        :param renyi_epsilon:
            The epsilon parameter in terms of Rényi differential privacy.
        :param desired_delta:
            The desired delta parameter for the approximate DP bound.
        """
        return ApproximatePrivacyLossBound(
            self.epsilon + (math.log(1 / desired_delta) / (self.order - 1)),
            desired_delta)

    def __str__(self):
        return f'({self.order}, {self.epsilon})-Rényi-DP'

    def __repr__(self):
        return f'RenyiPrivacyLossBound({self.order}, {self.epsilon})'


# Convert to dicts for JSON encoding for structured logging.
@logging_utils.encode.register(ApproximatePrivacyLossBound)
def _encode_approx_plb(arg: ApproximatePrivacyLossBound):
    return {
        'type': 'approx',
        'epsilon': arg.epsilon,
        'delta': arg.delta,
    }


@logging_utils.encode.register(PrivacyLossBound)
def _encode_plb(arg: PrivacyLossBound):
    return {
        'type': 'exact',
        'epsilon': arg.epsilon,
    }


@logging_utils.encode.register(RenyiPrivacyLossBound)
def _encode_renyi_plb(arg: RenyiPrivacyLossBound):
    return {
        'type': 'renyi',
        'order': arg.order,
        'epsilon': arg.epsilon,
    }
