# Copyright © 2023-2024 Apple Inc.

import itertools
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from pfl.common_types import Population


# Use eq=False on subclasses so that __eq__ and __hash__
# can be inherited from this base class:
# https://stackoverflow.com/a/53990477
@dataclass(frozen=True, order=True, eq=True)
class StringMetricName:
    """
    A structured name for metrics.

    :param description:
        The metric name represented as a string.
    """
    description: str

    def __str__(self) -> str:
        return self.description

    def __eq__(self, other):
        return isinstance(other, StringMetricName) and isinstance(
            self, StringMetricName) and str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


@dataclass(frozen=True, eq=False)
class MetricName(StringMetricName):
    """
    A structured name for metrics which includes the population the metric was
    generated from.

    :param description:
        The metric name represented as a string.
    :param population:
        The population the metric was generated from.
    """
    population: Population

    def _get_population_str(self):
        return (f'{self.population.value} population | ')

    def __str__(self) -> str:
        return f'{self._get_population_str()}{self.description}'


@dataclass(frozen=True, eq=False)
class TrainMetricName(MetricName):
    """
    A structured name for metrics which includes the population the metric was
    generated from and training descriptions.

    :param description:
        The metric name represented as a string.
    :param population:
        The population the metric was generated from.
    :param after_training:
        `True` if the metric was generated after local training, `False`
        otherwise.
    :param local_partition:
        (Optional) The name of the local dataset partition this metric is
        related to.
        This is mainly only relevant when you have a train and val set locally
        on devices.
    """
    after_training: bool
    local_partition: Optional[str] = None

    def __str__(self) -> str:
        partition = ('' if self.local_partition is None else
                     f'{self.local_partition} set | ')
        postfix = ' after local training' if self.after_training else ' before local training'

        return (f'{self._get_population_str()}{partition}'
                f'{self.description}{postfix}')


class ComposableMetricName(StringMetricName):
    """
    Base class of any decorate-able `MetricName`.

    An instance of ``ComposableMetricName`` can be composed with another
    instances of `ComposableMetricName` without losing the original attribute.

    The main purpose of this class and subclasses is to provide a way
    one can identify whether an attribute exist in a composed
    `ComposableMetricName`.

    :param metric_name:
        an instance of `MetricName`.
    """

    def __init__(self, metric_name: StringMetricName):
        assert isinstance(metric_name, StringMetricName)
        if self.__class__.__name__ == 'ComposableMetricName':
            raise NotImplementedError("Don't init this class")

        super().__init__(metric_name.description)
        # hacky, dataclasses doesn't support assignment of instance variables
        self._inner_metric_name = metric_name

    def __str__(self):
        return str(self._inner_metric_name)

    def __getattr__(self, key):
        """
        Recursively finds the key in all of composed ``ComposableMetricName``.
        """
        return getattr(self._inner_metric_name, key)


class MetricNamePostfix(ComposableMetricName):
    """
    Add a postfix to an existing metric name.

    :param metric_name:
        An instance of `MetricName`.
    :param postfix:
        The postfix to append to the metric name string.
    """

    def __init__(self, metric_name: StringMetricName, postfix: str):
        super().__init__(metric_name)
        self.postfix = postfix

    def __str__(self) -> str:
        inner_name = str(self._inner_metric_name)
        return f'{inner_name} | {self.postfix}'


class SkipSerialization(ComposableMetricName):
    """
    Metrics that shouldn't be serialized when consuming metrics.
    """

    @property
    def ignore_serialization(self):
        return True


class MetricValue(ABC):
    """
    The value of a metric, values of which that come from different places can
    be combined (using ``+``).
    Additionally, to allow for distributed computation, the value needs to be
    convertible to a fixed-dimensional vector of values.
    Addition in this space must perform the same operation as calling ``+`` on
    this object.

    Technically, any subclass ``MetricValue`` with ``+`` must form a commutative
    monoid: ``+`` must be associative and commutative.
    """

    @abstractmethod
    def __eq__(self, other):
        """
        Compare for equality.
        """

    @abstractmethod
    def __add__(self, other):
        """
        Combine two metric values of the same type.

        The order in which this is applied to a set of elements should not make
        a difference.
        """

    @property
    @abstractmethod
    def overall_value(self):
        """
        Return the overall value, e.g. an average or a total.
        """

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """
        Get a vector representation of this metric value, with
        ``dtype=float32``.
        Summing two vectors in this space must be equivalent to summing the two
        original objects.
        """
        pass

    @abstractmethod
    def from_vector(self, vector: np.ndarray) -> 'MetricValue':
        """
        Create a new metric value of this class from a vector representation.

        Note that this is a method on an object of this class, since it is
        possible that runtime attributes that do not change with addition are
        not serialized.
        """
        pass


"""
Metric values can be of type ``MetricValue``, or be a plain ``float`` or
``int``.
"""
MetricValueType = Union[MetricValue, float, int]


def user_average(metric_value: MetricValue) -> MetricValue:
    """
    Take the overall value of the provided metric and re-weight by 1.
    Can be used as a postprocess function when specifying metrics to be
    able to create average of per-user metrics instead of the metric over
    all data of the cohort.

    :param metric_value:
        Metric value to evaluate and re-weight by 1.
    :return:
        A new metric value with weight 1.
    """
    return Weighted(metric_value.overall_value, 1)


# These three functions forward to the matching methods on MetricValue, or if
# the argument is not of type MetricValue, perform the expected behaviour.


def get_overall_value(metric_value: MetricValueType) -> float:
    """
    Get the value of Union[MetricValue, float, int] in a type-safe way.
    """
    if isinstance(metric_value, MetricValue):
        return metric_value.overall_value
    return metric_value


def serialize_to_vector(metric_value: MetricValueType) -> np.ndarray:
    if isinstance(metric_value, MetricValue):
        return metric_value.to_vector()
    return np.asarray([metric_value], dtype=np.float32)


def deserialize_from_vector(metric_value: MetricValueType,
                            vector: np.ndarray) -> MetricValueType:
    if isinstance(metric_value, MetricValue):
        return metric_value.from_vector(vector)
    else:
        value = vector.item()
        return float(value)


class Weighted(MetricValue):
    """
    Represent a value with a weight.
    This allows a normalised value to be computed.
    These can be added; this adds the weighted values and the weights.

    E.g. the value could be the sum of single values, and the weight could be
    the number of values; then `value` is the weighted value.

    :param weighted_value:
        Value multiplied by the weight. E.g. if weighted value is 8.0 and
        weight is 2.0, the overall value is 4.0.
    :param weight:
        Weight for the value.
        If this is `0`, then the weighted value must be `0` too.
    """

    def __init__(self, weighted_value, weight):
        assert weight >= 0
        if weight == 0:
            assert weighted_value == 0
        self._weighted_value = float(weighted_value)
        self._weight = float(weight)

    @property
    def overall_value(self):
        if self._weight == 0:
            assert self._weighted_value == 0
            return 0.
        return self._weighted_value / float(self._weight)

    @property
    def weighted_value(self):
        return self._weighted_value

    @property
    def weight(self):
        return self._weight

    def __eq__(self, other):
        assert isinstance(other, Weighted)
        return (self._weighted_value
                == other._weighted_value) and (self.weight == other.weight)

    def __add__(self, other):
        assert isinstance(other, Weighted)
        return Weighted(self._weighted_value + other._weighted_value,
                        self.weight + other.weight)

    def __repr__(self):
        return f'({self._weighted_value}/{self._weight})'

    def from_vector(self, vector: np.ndarray) -> 'Weighted':
        return Weighted(*vector)

    def to_vector(self) -> np.ndarray:
        return np.array([self._weighted_value, self._weight], dtype=np.float32)

    """
    Construct from an unweighted value (e.g. an average).
    :param value:
        The unweighted value, which will be returned by `value()`.
    :param weight:
        (optional) The weight for this value.
        If not given, this will default to `1`.
    """

    @classmethod
    def from_unweighted(cls, value, weight=1):
        return cls(weight * value, weight)


class Summed(MetricValue):
    """
    A metric value which simply wraps 1 number. Addition is equal to summing.
    Unlike ``Weighted`` which accumulates a weighted average across devices,
    ``Summed`` is useful when the metric to be accumulated across devices
    should simply be the sum.

    :example:
        >>> assert Summed(1) + Summed(2) == Summed(3)
    """

    def __init__(self, value):
        self._value = value

    @property
    def overall_value(self):
        return self._value

    def __add__(self, other):
        return Summed(self._value + other._value)

    def __eq__(self, other):
        assert isinstance(other, Summed)
        return self._value == other._value

    def __repr__(self):
        return str(self._value)

    def from_vector(self, vector: np.ndarray) -> 'Summed':
        return Summed(*vector)

    def to_vector(self) -> np.ndarray:
        return np.array([self._value], dtype=np.float32)


class Histogram(MetricValue):
    """
    Collect values as a histogram. Use the ``from_values_range`` or
    ``from_values_bins`` to calculate the histogram from a list of
    values.

    :param bin_counts:
        A list of counts for each bin in the histogram.
    :param bins:
        A list of `N+1` boundaries representing the `N` bins.
    """

    def __init__(self, bin_counts: List[Number], bins: List[Number]):
        self._bin_counts = np.asarray(bin_counts)
        self._bins = np.asarray(bins)
        assert len(self._bin_counts) + 1 == len(self._bins)

    @classmethod
    def from_values_range(cls, values: List[float], num_bins: int,
                          min_bound: float, max_bound: float):
        """
        Create histogram from a list of values and equally-spaced
        bins.

        :param values:
            A list of values to create the histogram with.
        :param num_bins:
            Number of bins, equally-spaced between ``min_bound`` and
            ``max_bound``.
        :param min_bound:
            The leftmost bin edge. Values lower than this are ignored.
        :param max_bound:
            The rightmost bin edge. Values higher than this are ignored.
        """
        return cls(
            *np.histogram(values, bins=num_bins, range=(min_bound, max_bound)))

    @classmethod
    def from_values_bins(cls, values: List[float], bins: List[float]):
        """
        Create histogram from a list of values and custom bin edges,
        which don't need to be equally-spaced.

        :param values:
            A list of values to create the histogram with.
        :param bins:
            A list of `N+1` boundaries representing the `N` bins.
            If a value is outside the bin range, the value is ignored.
        """
        return cls(*np.histogram(values, bins=bins))

    @property
    def bin_counts(self):
        return self._bin_counts

    @property
    def bins(self):
        return self._bins

    @property
    def overall_value(self):
        raise NotImplementedError("property overall_value doesn't make "
                                  "sense for histogram metrics")

    def __eq__(self, other):
        assert isinstance(other, Histogram)
        return (all(np.equal(self._bins, other._bins))
                and all(np.equal(self._bin_counts, other._bin_counts)))

    def __add__(self, other):
        assert isinstance(other, Histogram)
        assert all(np.equal(self._bins, other._bins))
        return Histogram(self._bin_counts + other._bin_counts, self._bins)

    def __repr__(self):
        return f'hist(bin_counts={self._bin_counts}, bins={self._bins})'

    def from_vector(self, vector: np.ndarray) -> 'Histogram':
        return Histogram(vector, self._bins)

    def to_vector(self) -> np.ndarray:
        return self._bin_counts.astype(np.float32)


class MetricsZero:
    """
    Represent a set of metrics which has zero for all metric names.
    However, what the metric names are is not specified.
    This class only supports +.
    You probably want to use the singleton object `Zero`.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return '[0...]'

    def __str__(self):
        return '[0...]'

    def __add__(self, other):
        if isinstance(other, MetricsZero):
            return self
        elif isinstance(other, Metrics):
            return other
        else:
            return NotImplemented

    def __getitem__(self, metric_name: StringMetricName) -> MetricValueType:
        return NotImplemented


Zero = MetricsZero()


class Metrics:
    """
    Represent a set of metrics.

    It is possible to get a metric with `[]`.
    It is also possible to set it, but only if it does not exist yet.

    It is possible to iterate over the object, which yields tuples
    `(metric_name, value)`.

    It is possible to add two Metrics objects if they have the exact same set of
    metric names.
    It is possible to take the union (with `|`) of two Metrics objects if they
    have a completely disjunct set of metric names.

    It is possible to convert the metrics into a list of variable-length Numpy
    arrays with dtype ``float32`` and back.
    Adding these arrays is equivalent to adding the metrics.

    ``Metrics`` is built such that you can always get the value by specifying
    the key in string format, even if the actual key is a
    :class:`~pfl.metrics.StringMetricName`.

    :param metrics:
        (optional) An iterable of tuples `(metric_name, value)`
        to initialize the object with.
        This is equivalent to calling `self[metric_name] = value` for each
        entry.
        Use `from_dict` to initialize the class from a dict
        `{metric_name: value}`.
    """

    def __init__(self,
                 values: Optional[Iterable[Tuple[Union[str, StringMetricName],
                                                 MetricValueType]]] = None):
        if values is None:
            self._hash_to_keyvalue: Dict = {}
        else:
            self._hash_to_keyvalue = {
                hash(metric_name):
                (metric_name, typing.cast(MetricValueType, value))
                for metric_name, value in values
            }

    @classmethod
    def from_dict(cls, dic: Dict[Union[str, StringMetricName],
                                 MetricValueType]):
        return cls(list(dic.items()))

    def __len__(self) -> int:
        return len(self._hash_to_keyvalue)

    def __iter__(self):
        return iter(self._hash_to_keyvalue.values())

    def __contains__(self, metric_name: Union[str, StringMetricName]) -> bool:
        return hash(metric_name) in self._hash_to_keyvalue

    def __getitem__(
            self, metric_name: Union[str,
                                     StringMetricName]) -> MetricValueType:
        return self._hash_to_keyvalue[hash(metric_name)][1]

    def __setitem__(self, metric_name: Union[str, StringMetricName],
                    value: MetricValueType):
        name_hash = hash(metric_name)
        assert name_hash not in self._hash_to_keyvalue
        self._hash_to_keyvalue[name_hash] = (metric_name, value)

    def __repr__(self) -> str:
        return repr(self._hash_to_keyvalue)

    def __str__(self) -> str:
        # This will also invoke the string magic for the weighted values.
        return '{' + ", ".join(
            [f'{k}: {v}' for k, v in self._hash_to_keyvalue.values()]) + '}'

    def __add__(self, other) -> 'Metrics':
        if isinstance(other, MetricsZero):
            return self
        if set(self._hash_to_keyvalue) != set(other._hash_to_keyvalue):
            raise ValueError('Adding two "Metrics" objects that have different'
                             ' sets of metric names')
        return Metrics([(name, self[name] + other[name])
                        for name, _ in self._hash_to_keyvalue.values()])

    def __or__(self, other) -> 'Metrics':
        overlap = set(self._hash_to_keyvalue) & set(other._hash_to_keyvalue)
        if overlap != set():
            raise ValueError(f'Combining two "Metrics" objects that have '
                             f'{len(overlap)} overlapping names: {overlap}')

        return Metrics(itertools.chain(self, other))

    def to_simple_dict(
        self,
        force_serialize_all_metrics: bool = False
    ) -> Dict[str, Union[float, int]]:
        """
        Returns a python dictionary of name-value pairs of metrics and their
        values, e.g. {'Loss': 0.12, 'Accuracy': 0.45}. All metric names are
        capitalized.

        :param force_serialize_all_metrics:
            Default to False. Indicate whether or not to include metrics that
            are marked to be ignored on serialization.
        """

        def convert(metric_name, weighted_value):
            # Skip metric if it should not be sent.
            if not force_serialize_all_metrics:
                try:
                    if metric_name.ignore_serialization:
                        return None
                except AttributeError:
                    pass

            metric_name = str(metric_name)

            # Uppercase the first character.
            name_uppercase = metric_name[0].upper() + metric_name[1:]
            return (name_uppercase, get_overall_value(weighted_value))

        return dict(
            convert(*value) for value in self._hash_to_keyvalue.values()
            if convert(*value))

    def to_vectors(self) -> List[np.ndarray]:
        """
        Get a list of vector representations of the metric values in this
        object.

        :return:
            A list of ``np.ndarray``, which are vector representations of each
            metric value.
            The order of the vectors is the same for all `Metrics` objects with
            the same set of metric names.
            Performing element-wise addition of two of these vectors yields the
            same equivalent result of adding two
            :class:`~pfl.metrics.Metrics` objects
            (without the correctness checks).
        """
        return [
            serialize_to_vector(value)
            for _, value in sorted([(str(name), value)
                                    for name, value in self])
        ]

    def from_vectors(self, vectors: List[np.ndarray]) -> 'Metrics':

        def values():
            self_sorted = sorted([(str(name), name, value)
                                  for name, value in self])
            for ((_, name, value), vector) in zip(self_sorted, vectors):
                yield name, deserialize_from_vector(value, vector)

        return Metrics(values())
