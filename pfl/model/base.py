# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
A model contains all the functionality needed for simulating federated learning experiments with
the specific deep learning framework you have implemented your model with.
"""
from abc import abstractmethod
from typing import Callable, Generic, Optional, Tuple, TypeVar

from pfl.common_types import Saveable
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParamsType
from pfl.metrics import Metrics, StringMetricName
from pfl.stats import StatisticsType

ModelType = TypeVar('ModelType', bound='Model')
EvaluatableModelType = TypeVar('EvaluatableModelType',
                               bound='EvaluatableModel')
StatefulModelType = TypeVar('StatefulModelType', bound='StatefulModel')


class Model(Generic[StatisticsType]):
    """
    Model that can be trained with federated learning.

    Subclass this (or one of its base classes) to implement your own model.
    This class describes the minimal interface required for training the
    simplest algorithm possible.
    """

    @abstractmethod
    def apply_model_update(
            self: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """
        Compute updated parameters based on ``statistics``.

        This can either construct a new model and return it; or mutate ``self``
        and return ``self``.

        :param statistics:
            Statistics for updating the parameters of the model.

        :return:
            The new model and any metrics.
        """
        pass


class EvaluatableModel(Model, Generic[StatisticsType, ModelHyperParamsType]):

    @property
    def allows_distributed_evaluation(self) -> Optional[bool]:
        """
        Distributed evaluation can only be enabled when,
        whether or not splitting the dataset, doing evaluation on the
        separate datasets and summing the metrics ends up with the same results
        as doing one call to evaluate with all data.

        As a conservative setting, distributed evaluation is not allowed by
        default. Every model subclass has to explicitly make sure distributing
        evaluation will work correctly.
        If set to `None`, it is interpreted as not yet determined, and any
        evaluation that support distributed evaluation will not be
        distributed.
        """
        return False

    @abstractmethod
    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn: Callable[[str], StringMetricName],
            eval_params: Optional[ModelHyperParamsType] = None) -> Metrics:
        """
        Evaluate performance of model on the given input data.

        This can be used in different circumstances.
        One is for simulated distributed evaluation, where the data is supposed
        to be from one device.
        Another is for centrally held data.

        :param dataset:
            Dataset to evaluate on.
            If this is centrally held data, it is still a flat list of data
            points.
        :param name_formatting_fn:
            A function to be used to generate a metric name object from a
            simple string, which will adorn the string with additional
            information.
        :param eval_params:
            Optional model parameters to use for evaluating the models. Some
            models can evaluate without a parameter object.

        :returns:
            A `Metrics` object with performance metrics.
        """


class StatefulModel(EvaluatableModel[StatisticsType, ModelHyperParamsType],
                    Saveable):
    """
    Convenience class for a model that has a fixed number of parameters
    and is stateful.

    (This is true for frameworks for neural networks in a federated setting,
    hence the eclectic set of requirements.)

    Simulation of local training in this setup is done by mutating the
    parameters in four steps:

    1. backing up the current (central) parameters (``get_parameters``)

    2. training the inner model on a user's local data.

    3. computing the difference between the current parameters and the
       backed-up ones (``get_model_difference``)

    4. restoring to the backed-up state (``set_parameters``).
    """

    @abstractmethod
    def get_model_difference(self,
                             other_parameters: StatisticsType,
                             clone: bool = False) -> StatisticsType:
        """
        Get the model difference between the current state of the model and
        the other state given as input (i.e. current-other).

        :param other_parameters:
            Get the model difference with respect to this other state of
            model parameters. Can be received from ``get_parameters``.
        :param clone:
            If ``False``, there is a chance that the implementation can use a
            cache to make this method run faster. This is fine as long as you
            don't require to hold multiple difference statistics in memory at
            the same time. If you do need to e.g.

            .. code-block:: python

              diff1 = model.get_model_difference(initial_params, clone=True)
              diff2 = model.get_model_difference(initial_params, clone=True)

            then ``clone`` should be ``True`` because otherwise there is a
            chance that diff1 points to diff2 after the second call.

        :returns:
            The model difference statistics.
        """

    @abstractmethod
    def get_parameters(
            self,
            placeholders: Optional[StatisticsType] = None) -> StatisticsType:
        """
        Retrieve model parameters at the current state.
        Useful if you want to restore the model to this
        state after it has been modified.

        :param placeholders:
            Statistics with tensors to be updated in-place
            with model parameters. If the model supports this,
            it will greatly increase performance when training
            many users.
        """

    @abstractmethod
    def set_parameters(self, w: StatisticsType) -> None:
        """
        Apply new model parameters to the model. This is
        used to restore the model to a previous state,
        which can be received from ``get_parameters``.
        """
