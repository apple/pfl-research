# Copyright © 2023-2024 Apple Inc.
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Protocol, Tuple, Type, TypeVar, Union

import numpy as np

from pfl.internal.ops.selector import get_default_framework_module as get_ops

# Need to disable pytype "invalid-annotation" because there are false positives:
# https://github.com/google/pytype/issues/379
# https://github.com/google/pytype/issues/704
# pytype: disable=invalid-annotation


class TensorLike(Protocol):
    """
    A protocol for tensor-like objects, e.g. NumPy,
    TensorFlow and Torch tensors.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        ...

    def __iter__(self) -> Iterator:
        raise NotImplementedError

    @property
    def shape(self) -> Tuple:
        ...


Tensor = TypeVar('Tensor', bound=TensorLike)
StatisticsType = TypeVar('StatisticsType', bound='TrainingStatistics')
MappedVectorStatisticsType = TypeVar('MappedVectorStatisticsType',
                                     bound='MappedVectorStatistics')


class TrainingStatistics(ABC, Generic[Tensor]):
    """
    Base class for statistics that can be used for training a model.
    Statistics from different parts of the data can be combined with "+".

    The values can have different types depending on the subclass.

    Statistics can be converted to vector space,
    summed, and then converted back, which should be equivalent to
    summing the actual statistics.

    :example:

        .. code-block:: python

            metadata1, private_vectors1 = stats1.get_weights()
            metadata2, private_vectors2 = stats2.get_weights()
            metadata_sum = metadata1 + metadata2
            private_vectors_sum = [
                v1 + v2 for v1, v2 in zip(private_vectors1,
                                          private_vectors2)
            ]
            stats_sum = stats1.from_weights(metadata_sum,
                                            private_vectors_sum)
            assert stats_sum == (stats1 + stats2)
    """

    @abstractmethod
    def __add__(self: StatisticsType, other: StatisticsType) -> StatisticsType:
        """
        Combine two compatible sets of training statistics, in such a way that
        the optimisation of the parameters based on the statistics will perform
        the optimisation on the union of the two datasets that the two sets
        of statistics were computed on.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Logging-friendly representation of statistics.
        """
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """
        Get the total number of parameters for all statistics in this object.
        """
        pass

    @abstractmethod
    def get_weights(self) -> Tuple[Tensor, List[Tensor]]:
        """
        Get a vector representation of this statistic, with ``dtype=float32``.
        Usually for the purpose of sending it over the wire. The vector
        representation returned may be divided up into smaller vector/matrices
        to more easily be manipulated.

        Summing vectors returned by ``get_weights`` from two different
        statistics objects and thereafter converting back using the method
        ``from_weights`` must be equivalent to summing the two original
        objects.

        :returns:
            A tuple `(metadata, weights)`. `metadata` is a vector
            of data which is not privacy-sensitive, e.g. the weight.
            `weights` is a list of matrices with data that is
            considered privacy-sensitive and is additive with noise.
            The matrices may be weights for different layers in the context
            of neural networks.
        """
        pass

    @abstractmethod
    def from_weights(self: StatisticsType, metadata: Tensor,
                     weights: List[Tensor]) -> StatisticsType:
        """
        Create a new statistics of this class from a vector representation.
        The input of this method is the same format as returned by
        ``get_weights``.

        Note that this is a method on an object of this class, since it is
        possible that runtime attributes that do not change with addition are
        not serialized.

        :param metadata:
            A vector of data which is not privacy-sensitive. The contents
            depend on the implementation. Can include e.g. the weight.
        :param weights:
            A list of matrices with data that is considered privacy-sensitive
            and is additive with noise. The matrices may be weights for
            different layers in the context of neural networks.
        """
        pass

    # Type hints for Callable to accept *args and **kwargs not possible yet:
    # https://github.com/python/mypy/issues/5876
    def apply(self: StatisticsType, fn: Callable[..., Iterable[np.ndarray]],
              *args, **kwargs) -> StatisticsType:
        """
        Apply a function on the weights from `get_weights`, and put result
        back into a new statistics with same metadata.

        :example:

            .. code-block:: python

                stats_p1 = stats.apply(lambda weights: [w+1 for w in weights])


        :param fn:
            A callable `(weights, *args, **kwargs) -> weights`, where `weights`
            is a list of tensors and `args` and `kwargs` are any additional
            arguments.
        :param args:
            Additional arguments to `fn` when applying it.
        :param kwargs:
            Additional keyword arguments to `fn` when applying it.
        :return:
            A statistics, with its data transformed by `fn`.
        """
        metadata, weights = self.get_weights()
        transformed_weights = fn(weights, *args, **kwargs)
        return self.from_weights(metadata, list(transformed_weights))

    # Type hints for Callable to accept *args and **kwargs not possible yet:
    # https://github.com/python/mypy/issues/5876
    def apply_elementwise(self: StatisticsType, fn: Callable[..., np.ndarray],
                          *args, **kwargs) -> StatisticsType:
        """
        Apply function on each weight from `get_weights` individually,
        and put result back into a new statistics with same metadata.

        :example:

            .. code-block:: python

                # Equivalent to the example in `apply`.
                stats_p1 = stats.apply_elementwise(lambda w: w+1)


        :param fn:
            A callable `(weight, *args, **kwargs) -> weight`, where `weight`
            is a tensor from the statistic's weights and `args` and `kwargs`
            are any additional arguments.
        :param args:
            Additional arguments to `fn` when applying it.
        :param kwargs:
            Additional keyword arguments to `fn` when applying it.
        :return:
            A statistics, with its data transformed by `fn`.
        """
        return self.apply(
            lambda tensors: [fn(t, *args, **kwargs) for t in tensors])


class WeightedStatistics(TrainingStatistics[Tensor]):
    """
    Statistics for training a model that can be weighted and summed.
    The weight will generally be the number of samples or the number of clients
    that the statistics are over.
    Using the method ``average`` produces a weighted average of the summed
    statistics.

    The statistics can be re-weighted by the `weight` property.

    In mathematical terms, the statistics are in a vector space; and with the
    weights, they are in an expectation semiring.

    :param weight:
        The weight of the statistics object.
    """

    def __init__(self, weight: float):
        self._weight = weight

    @property
    def weight(self) -> float:
        """
        Get the weight of this object.
        """
        return self._weight

    @weight.setter
    def weight(self, value) -> None:
        """
        Set the weight of this object.
        Does not affect the values of the statistics.
        To change the weight without affecting ratio statistics/weight,
        use ``reweight`` method.
        """
        self._weight = value

    @abstractmethod
    def reweight(self, new_weight: float):
        """
        Reweight the statistics by dividing the values by the current
        weight and multiplying them by ``new_weight``. This means that
        the ratio statistics/weight remains the same but the weight changes
        to ``new_weight``.
        """
        pass

    def average(self) -> None:
        """
        Divide (in-place) each individual statistic by its weight.
        The new weight will be ``1``.
        """
        self.reweight(1)


class MappedVectorStatistics(WeightedStatistics[Tensor]):
    """
    Statistics consisting of a number of tensors keyed by strings.
    Commonly used to represent neural network model updates. When adding
    two ``Statistics``, the tensors for each key are added together
    and the weights are added as well.

    :param name_to_stats:
        A dictionary, mapping identifiers of individual statistics to the
        tensors.
    :param weight:
        The weight of the statistics. Does not have any effect on the raw data
        directly.
    """

    def __init__(self,
                 name_to_stats: Optional[Dict[str, Tensor]] = None,
                 weight: float = 1.0):
        self._data = name_to_stats or {}
        super().__init__(weight)

    @property
    def num_parameters(self) -> int:
        return sum(np.prod(get_ops().get_shape(t)) for t in self.values())

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"MappedVectorStatistics(weight={self.weight} data={self._data})"

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        """
        Set the statistic of a given key.
        Note that these are pre-weighted statistics.
        You should therefore multiply the statistics value by the weight before
        setting it.
        """
        self._data[key] = value

    def __add__(
            self: MappedVectorStatisticsType,
            other: MappedVectorStatisticsType) -> MappedVectorStatisticsType:
        """
        Add two ``MappedVectorStatistics`` by adding their individual vectors.
        The weights are also added.
        """
        assert set(self._data.keys()) == set(
            other.keys()), "Statistics objects needs to have the same keys."
        added_raw_stats = {k: self[k] + other[k] for k in self._data}
        stats: MappedVectorStatisticsType = self.__class__(added_raw_stats,
                                                           weight=self.weight +
                                                           other.weight)
        return stats

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def reweight(self, new_weight: float):
        # pylint: disable=access-member-before-definition
        multiplier = float(new_weight / self.weight)
        for name in self.keys():
            self[name] *= multiplier
        self.weight = new_weight
        # pylint: enable=access-member-before-definition

    def get_weights(self) -> Tuple[Tensor, List[Tensor]]:
        return get_ops().to_tensor([self.weight]), list(self.values())

    def from_weights(self: MappedVectorStatisticsType, metadata: Tensor,
                     weights: List[Tensor]) -> MappedVectorStatisticsType:
        assert len(metadata) == 1
        weight_tensor: Tensor = metadata[0]
        weight: float = get_ops().to_numpy(weight_tensor).item()

        vector_mapping = dict(zip(self.keys(), weights))
        stats: MappedVectorStatisticsType = self.__class__(
            vector_mapping, weight)
        return stats

    def pop(self, key: str) -> Tuple[Tensor, Tensor]:
        return self._data.pop(key), get_ops().to_tensor(self._weight)

    @classmethod
    def from_vector(
            cls: Type[MappedVectorStatisticsType], vector: Tensor,
            weight: Union[float, Tensor], names: List[str],
            shapes: List[Tuple[int, ...]]) -> MappedVectorStatisticsType:
        assert isinstance(weight, float)
        vector = get_ops().to_tensor(vector)
        values = get_ops().reshape(vector, shapes)
        assert len(names) == len(values)
        return cls(dict(zip(names, values)), weight)


class ElementWeightedMappedVectorStatistics(MappedVectorStatistics[Tensor]):
    """
    Statistics consisting of a number of tensors keyed by strings with weights
    as a number of tensors keyed by the same set of strings. Each element in the
    statistics has a weight tensor with the same shape and the same key as the
    element.
    Commonly used to represent neural network model updates.

    :param name_to_stats:
        A dictionary, mapping identifiers of individual statistics to the
        tensors.
    :param weights:
        The dictionary with weights of the statistics. Does not have any effect
        on the raw data directly.
        Adding two ``Statistics`` will add their weights as well.
    """

    def __init__(self,
                 name_to_stats: Optional[Dict[str, Tensor]] = None,
                 weights: Optional[Dict[str, Tensor]] = None):
        self._data = name_to_stats or {}
        self._weights = weights or {}
        if not self._weights:
            for key in self._data:
                self._weights[key] = get_ops().to_tensor(
                    np.ones_like(self._data[key]))
        else:
            self._validate_keys_and_shapes(self._data, self._weights)

    def _validate_keys_and_shapes(self, this: Dict[str, Tensor],
                                  other: Dict[str, Tensor]) -> None:
        """
        Validate if two tensor dictionaries have same keys and matched shapes.
        """
        assert set(this.keys()) == set(
            other.keys()), "Tensor dictionaries need to have the same keys."
        for key in this:
            assert this[key].shape == other[key].shape, (
                f'Shape mismatch for key {key}! Shape: {this[key].shape} v.s. Shape: {other[key].shape}'
            )

    def __repr__(self):
        return f"ElementWeightedMappedVectorStatistics(weights={self.weights} data={self._data})"

    @property
    def weight(self) -> float:
        raise NotImplementedError

    @weight.setter
    def weight(self, value) -> None:
        raise NotImplementedError

    def reweight(self, new_weight: float):
        raise NotImplementedError

    @property
    def weights(self) -> Dict[str, Tensor]:
        return self._weights

    @weights.setter
    def weights(self, value) -> None:
        """
        Set the weight vector of this object.
        Does not affect the values of the statistics.

        Overriding ``MappedVectorStatistics.weight.setter``.
        """
        self._validate_keys_and_shapes(self._weights, value)
        self._weights = value

    def __add__(
        self: 'ElementWeightedMappedVectorStatistics',
        other: 'ElementWeightedMappedVectorStatistics'
    ) -> 'ElementWeightedMappedVectorStatistics':
        """
        Add two ``ElementWeightedMappedVectorStatistics`` by adding their
        individual vectors.
        The weights are also added.
        """
        assert set(self._data.keys()) == set(
            other.keys()), "Statistics objects needs to have the same keys."
        assert set(self._weights.keys()) == set(other.weights.keys(
        )), "Statistics weights needs to have the same keys."
        added_weight = {
            k: self._weights[k] + other.weights[k]
            for k in self._weights
        }
        added_raw_stats = {k: self[k] + other[k] for k in self._data}
        return ElementWeightedMappedVectorStatistics(added_raw_stats,
                                                     weights=added_weight)

    def get_weights(self) -> Tuple[Tensor, List[Tensor]]:
        weights = get_ops().flatten([self._weights[k] for k in self.keys()])[0]
        return weights, list(self.values())

    def from_weights(
        self, metadata: Tensor, statistics_list: List[Tensor]
    ) -> 'ElementWeightedMappedVectorStatistics':
        flattened_weight = metadata
        assert len(flattened_weight.shape) > 0
        weight_values = get_ops().reshape(
            flattened_weight, [self.weights[k].shape for k in self.keys()])
        weight_mapping = dict(zip(self.keys(), weight_values))
        assert len(self) == len(statistics_list)
        vector_mapping = dict(zip(self.keys(), statistics_list))

        return ElementWeightedMappedVectorStatistics(vector_mapping,
                                                     weight_mapping)

    def average(self) -> None:
        new_weight = {
            k: get_ops().to_tensor(np.ones_like(self[k]))
            for k in self.keys()
        }
        for name in self.keys():
            self[name] *= new_weight[name] / self.weights[name]
        self.weights = new_weight

    def pop(self, key: str) -> Tuple[Tensor, Tensor]:
        return self._data.pop(key), self._weights.pop(key)

    @classmethod
    def from_vector(
        cls, statistics_vector: Tensor, weights: Union[float, Tensor],
        names: List[str],
        shapes: List[Tuple[int,
                           ...]]) -> 'ElementWeightedMappedVectorStatistics':
        statistics_vector = get_ops().to_tensor(statistics_vector)
        weights = get_ops().to_tensor(weights)
        data_values = get_ops().reshape(statistics_vector, shapes)
        weight_values = get_ops().reshape(weights, shapes)
        assert len(names) == len(data_values)
        assert len(names) == len(weight_values)
        return cls(dict(zip(names, data_values)),
                   dict(zip(names, weight_values)))
