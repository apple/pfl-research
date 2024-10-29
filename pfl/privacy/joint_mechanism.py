# Copyright © 2023-2024 Apple Inc.
'''
Joint mechanism for combining multiple mechanisms into one.
'''

from typing import Dict, List, Optional, Set, Tuple

from pfl.hyperparam import HyperParamClsOrFloat, get_param_value
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import MappedVectorStatistics, TrainingStatistics

from . import compute_parameters
from .approximate_mechanism import SquaredErrorLocalPrivacyMechanism
from .privacy_accountant import PrivacyAccountant
from .privacy_mechanism import CentrallyApplicablePrivacyMechanism, PrivacyMechanism
from .privacy_snr import SNRMetric


def check_if_partition(full_set: Set[str], partition: List[Set[str]]):
    """
    Checks if each element of full_list appears in exactly one of
    the sets in partition
    """
    partition_union: Set = set()
    partition_size = 0
    for s in partition:
        partition_union = partition_union.union(s)
        partition_size += len(s)

    return (partition_union == full_set) and (partition_size == len(full_set))


class JointMechanism(CentrallyApplicablePrivacyMechanism):
    """
    Constructs a new CentrallyApplicablePrivacyMechanism from existing ones.
    Each existing mechanism is applied to a disjoint subset of the client
    statistics keys. As such JointMechanism can only be applied to client
    statistics of type MappedVectorStatistics.

    :param mechanisms_and_keys:
        Dictionary in which each key is a name of a mechanism and each value
        is a tuple consisting of the corresponding CentrallyApplicablePrivacyMechanism
        and a list of keys specifying which portion of the user training statistics that
        mechanism should be applied to. Note the names of each of the mechanisms must be
        distinct for the purpose of naming the corresponding Metrics.

    """

    def __init__(
        self,
        mechanisms_and_keys: Dict[str,
                                  Tuple[CentrallyApplicablePrivacyMechanism,
                                        List[str]]]):
        if len(set(mechanisms_and_keys.keys())) < len(
                mechanisms_and_keys.keys()):
            raise ValueError('Mechanism names must be unique.')
        self.mechanisms_and_keys = mechanisms_and_keys

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        if not check_if_partition(
                set(statistics.keys()),
            [set(keys) for _, keys in self.mechanisms_and_keys.values()]):
            raise ValueError(
                'Mechanism keys do not form a partition of the client statistics keys.'
            )

        if not isinstance(statistics, MappedVectorStatistics):
            raise ValueError(
                'Statistics must be of type MappedVectorStatistics.')

        metrics = Metrics()
        for mechanism_name, (
                mechanism,
                statistics_keys) in self.mechanisms_and_keys.items():

            def mechanism_name_formatting_fn(n, prefix=mechanism_name):
                return name_formatting_fn(f'{prefix}: {n}')

            sub_statistics: MappedVectorStatistics = MappedVectorStatistics()
            for key in statistics_keys:
                sub_statistics[key] = statistics[key]
            clipped_sub_statistics, sub_metrics = mechanism.constrain_sensitivity(
                sub_statistics, mechanism_name_formatting_fn, seed)
            for key in statistics_keys:
                statistics[key] = clipped_sub_statistics[key]
            metrics = metrics | sub_metrics

        return statistics, metrics

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        if not check_if_partition(
                set(statistics.keys()),
            [set(keys) for _, keys in self.mechanisms_and_keys.values()]):
            raise ValueError(
                'Mechanism keys do not form a partition of the client statistics keys.'
            )

        if not isinstance(statistics, MappedVectorStatistics):
            raise ValueError(
                'Statistics must be of type MappedVectorStatistics.')

        metrics = Metrics()
        for mechanism_name, (
                mechanism,
                statistics_keys) in self.mechanisms_and_keys.items():

            def mechanism_name_formatting_fn(n, prefix=mechanism_name):
                return name_formatting_fn(f'{prefix}: {n}')

            sub_statistics: MappedVectorStatistics = MappedVectorStatistics()
            for key in statistics_keys:
                sub_statistics[key] = statistics[key]
            noised_sub_statistics, sub_metrics = mechanism.add_noise(
                sub_statistics, cohort_size, mechanism_name_formatting_fn,
                seed)
            for key in statistics_keys:
                statistics[key] = noised_sub_statistics[key]
            metrics = metrics | sub_metrics

        return statistics, metrics
