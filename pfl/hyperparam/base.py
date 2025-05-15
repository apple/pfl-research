# Copyright © 2023-2024 Apple Inc.

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Generic, Optional, TypeVar, Union

ParameterType = TypeVar('ParameterType')
AlgorithmHyperParamsType = TypeVar('AlgorithmHyperParamsType',
                                   bound='AlgorithmHyperParams')
ModelHyperParamsType = TypeVar('ModelHyperParamsType',
                               bound='ModelHyperParams')


class HyperParam(ABC, Generic[ParameterType]):
    """
    Base class for defining adaptive hyperparameters.
    An adaptive hyperparameter can be used as a substitute for a static
    parameter where permitted (mostly in the configs and algorithms, see
    respective type signatures).

    Make the subclass also a postprocessor or callback to access hooks
    where adapting the hyperparameters can take place.

    :example:
        This is an example of an adaptive hyperparameter (cohort size),
        which increases by a factor of 2 each central iteration.

        .. code-block:: python

            class MyCohortSize(HyperParam, TrainingProcessCallback):
                def __init__(self, initial):
                    self._value = initial

                def after_central_iteration(self, aggregate_metrics, model,
                                            central_iteration):
                    self._value *= 2

                def value(self):
                    return self._value
    """

    @abstractmethod
    def value(self) -> ParameterType:
        """
        The current state (inner value) of the hyperparameter.
        """
        pass


# Type aliases for maybe HyperParameter or its inner type.
HyperParamClsOrBool = Union[HyperParam[bool], bool]
HyperParamClsOrInt = Union[HyperParam[int], int]
HyperParamClsOrFloat = Union[HyperParam[float], float]


def get_param_value(parameter):
    """
    If input is a :class:`~pfl.hyperparam.base.HyperParam`,
    extract its current value, otherwise return identity.

    :example:

      .. code-block:: python

        >>> get_param_value(1.0)
        1.0
        >>> get_param_value(MyCustomParam(initial_value=2.0))
        2.0
    """
    if isinstance(parameter, HyperParam):
        return parameter.value()
    elif isinstance(parameter, HyperParams):
        return parameter.static_clone()
    else:
        return parameter


# WARNING! We should avoid putting default arguments in dataclasses which can be
# subclassed because of this bug https://bugs.python.org/issue36077
# which has a (open as of 2020-06-28) PR here
# https://github.com/python/cpython/pull/17322.
# Update: fix included in Python 3.10
# (https://github.com/python/cpython/pull/25608) but we can't include it until
# we remove <=3.10 compatibility.

HyperParamsType = TypeVar('HyperParamsType', bound='HyperParams')


@dataclass(frozen=True)
class HyperParams:
    """
    Base class for dataclasses that store parameters for model/training.
    """

    def __post_init__(self):
        super().__init__()

    def static_clone(self: HyperParamsType, **kwargs) -> HyperParamsType:
        """
        Returns a static clone of hyperparameters where each parameter has
        its current value (including adaptive parameters). This is used
        to access parameters in the algorithms (e.g. ``FederatedNNAlgorithm``).
        """
        current_params = self.to_context_dict()
        return self.__class__(**{**dict(current_params), **kwargs})

    def get(self, key):
        """
        Get the current static value of a hyperparameter (which is a property
        of the dataclass). I.e. in the case of the hyperparam being a
        :class:`~pfl.hyperparam.base.HyperParam`, return the inner value
        state.
        """
        return get_param_value(getattr(self, key))

    def to_context_dict(self) -> Dict[str, Any]:
        return {
            f.name: get_param_value(getattr(self, f.name))
            for f in fields(self)
        }


@dataclass(frozen=True)
class AlgorithmHyperParams(HyperParams):
    """
    Base class for additional parameters to pass to algorithms.
    By default, this base class has no parameters, but subclasses
    purposed for certain federated algorithms will have additional
    parameters.
    """
    pass


@dataclass(frozen=True)
class ModelHyperParams(HyperParams):
    """
    A base configuration for training models.
    By default, this base class has no parameters, but subclasses
    purposed for certain models will have additional
    parameters.
    """


@dataclass(frozen=True)
class NNEvalHyperParams(ModelHyperParams):
    """
    Config to use for evaluating any neural network with an algorithm
    that involves SGD.

    :param local_batch_size:
        The batch size for evaluating locally on device.
        If `None`, defaults to no batching (full-batch evaluation).
    """
    local_batch_size: Optional[HyperParamClsOrInt]


@dataclass(frozen=True)
class NNTrainHyperParams(NNEvalHyperParams):
    """
    Config to use for training any neural network with an algorithm that
    involves SGD.

    :param local_num_epochs:
        The number of epochs of training applied on the device.
        If this is set, ``local_num_steps`` must be ``None``.
    :param local_learning_rate:
        The learning rate applied on the device.
    :param local_batch_size:
        The batch size for training locally on device.
        If `None`, defaults to the entire dataset, which means one local
        iteration is one epoch.
    :param local_max_grad_norm:
        Maximum L2 norm for gradient update in each local optimization step.
        Local gradients on device will be clipped if their L2 norm is larger
        than `local_max_grad_norm`. If `None`, no clipping is performed.
    :param local_num_steps:
        Number of gradient steps during local training. If this is set,
        ``local_num_epochs`` must be ``None``. Stops before
        ``local_num_steps`` if iterated through dataset.
        This can be useful if the user dataset is very large and training
        less than an epoch is appropriate.
    :param grad_accumulation_steps:
        Number of steps to accumulate gradients before apply a local optimizer
        update. The effective batch size is ``local_batch_size`` multiplied by
        this number. This is useful to simulate a larger local batch size when
        memory is limited. Currently only supported for PyTorch.
    """
    local_num_epochs: Optional[HyperParamClsOrInt]
    local_learning_rate: HyperParamClsOrFloat
    local_max_grad_norm: Optional[HyperParamClsOrFloat] = None
    local_num_steps: Optional[HyperParamClsOrInt] = None
    grad_accumulation_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.local_max_grad_norm is not None:
            assert get_param_value(self.local_max_grad_norm) > 0., (
                "Local max gradient norm for clipping must be positive")

        if not ((self.local_num_steps is None) ^
                (self.local_num_epochs is None)):
            raise ValueError("Either local_num_steps or local_num_epochs "
                             "must be set, but not both")
