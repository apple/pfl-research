# Copyright © 2023-2024 Apple Inc.
'''
Joint mechanism for combining multiple mechanisms into one.
'''

from typing import Dict, List, Optional, Set, Tuple

from pfl.metrics import Metrics, StringMetricName
from pfl.stats import MappedVectorStatistics, TrainingStatistics

from .privacy_mechanism import CentrallyApplicablePrivacyMechanism


class JointMechanism(CentrallyApplicablePrivacyMechanism):
    """
    Constructs a new CentrallyApplicablePrivacyMechanism from existing ones.
    Each existing mechanism is applied to a disjoint subset of the client
    statistics keys. As such JointMechanism can only be applied to client
    statistics of type MappedVectorStatistics.

    :param mechanisms_and_keys:
        Dictionary which maps a name of a mechanism to a tuple consisting of
        the corresponding CentrallyApplicablePrivacyMechanism and a list specifying
        which keys of the user training statistics that mechanism should be applied
        to. This list can contain strings of the following forms:

        * key_name - an exact key name as appearing in the user statistics,
        * f'{key_prefix}/' - matches to all user statistics keys of the form
        f'{key_prefix}/{any_string}'.

        Finally, note that the names of each of the mechanisms must be distinct
        for the purpose of naming the corresponding Metrics.

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

        if not isinstance(statistics, MappedVectorStatistics):
            raise TypeError(
                'Statistics must be of type MappedVectorStatistics.')

        clipped_statistics: MappedVectorStatistics = MappedVectorStatistics(
            weight=statistics.weight)
        metrics = Metrics()
        client_statistics_keys = set(statistics.keys())
        for mechanism_name, (
                mechanism, mechanism_keys) in self.mechanisms_and_keys.items():

            def mechanism_name_formatting_fn(n, prefix=mechanism_name):
                return name_formatting_fn(f'{prefix} | {n}')

            # Extract client statistics keys that match the keys for current mechanism
            sub_statistics: MappedVectorStatistics = MappedVectorStatistics()
            for key in mechanism_keys:
                if key in client_statistics_keys:  # exact key name
                    sub_statistics[key] = statistics[key]
                    client_statistics_keys.remove(key)
                else:
                    assert key[
                        -1] == '/', f"{key} does not appear as a key in the client statistics."
                    for client_key in statistics:
                        if client_key.startswith(
                                key):  # matches f'{key_prefix}/'
                            sub_statistics[client_key] = statistics[client_key]
                            client_statistics_keys.discard(client_key)

            # Clip statistics using mechanism
            clipped_sub_statistics, sub_metrics = mechanism.constrain_sensitivity(
                sub_statistics, mechanism_name_formatting_fn, seed)

            # Recombine clipped statistics and metrics
            for key in clipped_sub_statistics:
                clipped_statistics[key] = clipped_sub_statistics[key]
            metrics = metrics | sub_metrics

        if len(client_statistics_keys) > 0:
            raise ValueError(
                f'Not all client statistics have been clipped. '
                f'These keys are missing from mechanisms_and_keys: {client_statistics_keys}.'
            )

        return clipped_statistics, metrics

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        if not isinstance(statistics, MappedVectorStatistics):
            raise TypeError(
                'Statistics must be of type MappedVectorStatistics.')

        noised_statistics: MappedVectorStatistics = MappedVectorStatistics(
            weight=statistics.weight)
        metrics = Metrics()
        client_statistics_keys = set(statistics.keys())
        for mechanism_name, (
                mechanism, mechanism_keys) in self.mechanisms_and_keys.items():

            def mechanism_name_formatting_fn(n, prefix=mechanism_name):
                return name_formatting_fn(f'{prefix} | {n}')

            # Extract client statistics keys that match the keys for current mechanism
            sub_statistics: MappedVectorStatistics = MappedVectorStatistics()
            for key in mechanism_keys:
                if key in client_statistics_keys:  # exact key name
                    sub_statistics[key] = statistics[key]
                    client_statistics_keys.remove(key)
                else:
                    assert key[
                        -1] == '/', f"{key} does not appear as a key in the client statistics."
                    for client_key in statistics:
                        if client_key.startswith(
                                key):  # matches f'{key_prefix}/'
                            sub_statistics[client_key] = statistics[client_key]
                            client_statistics_keys.discard(client_key)

            # Apply noise using mechanism
            noised_sub_statistics, sub_metrics = mechanism.add_noise(
                sub_statistics, cohort_size, mechanism_name_formatting_fn,
                seed)

            # Recombine noised statistics and metrics
            for key in noised_sub_statistics:
                noised_statistics[key] = noised_sub_statistics[key]
            metrics = metrics | sub_metrics

        if len(client_statistics_keys) > 0:
            raise ValueError(
                f'Not all client statistics have been noised. '
                f'These keys are missing from mechanisms_and_keys: {client_statistics_keys}.'
            )

        return noised_statistics, metrics
