# Copyright © 2023-2024 Apple Inc.
"""
Algorithms of base class :class:`~pfl.algorithm.base.FederatedAlgorithm` implement local training of models and processing of central model updates. It is these two parts of the end-to-end training loop that define the behaviour of a specific federated algorithm. The remaining parts define the private federated learning framework itself and do not change with different training algorithms.
"""
import json
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np

from pfl.aggregate.base import Backend
from pfl.callback import TrainingProcessCallback
from pfl.common_types import Population, Saveable
from pfl.context import CentralContext
from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam.base import (
    AlgorithmHyperParams,
    AlgorithmHyperParamsType,
    HyperParamClsOrInt,
    ModelHyperParamsType,
    get_param_value,
)
from pfl.internal.platform.selector import get_platform
from pfl.metrics import Metrics, TrainMetricName
from pfl.model.base import ModelType, StatefulModelType
from pfl.stats import StatisticsType

from . import algorithm_utils

logger = logging.getLogger(__name__)


class FederatedAlgorithm(Saveable,
                         Generic[AlgorithmHyperParamsType,
                                 ModelHyperParamsType, ModelType,
                                 StatisticsType, AbstractDatasetType]):
    """
    Base class for federated algorithms.

    Federated algorithms consist of a computation on multiple clients and
    a computation on the server that processes the results from the clients.
    This class is where all pieces of computation can be implemented.
    It enforces a specific structure so that it is possible to switch between
    simulation and live training with minimal changes.

    A subclass of ``FederatedAlgorithm`` should provide
    ``process_aggregated_statistics``, which is the server-side part of the
    computation.
    In addition, it is useful to run simulations with client-side computation
    locally.
    To do this, ``simulate_one_user`` can implement the client-side computation
    in this same class.
    Finally, the ``run`` method performs the orchestration.

    The object connecting the server-side and the client-side computation is
    the backend, passed into ``run``.
    In simulation, this may be a :class:`~pfl.aggregate.SimulatedBackend`
    which calls ``simulate_one_user`` to perform simulation,
    In live training, the backend will call out to real devices.

    When subclassing, new parameters can either go into constructor or by
    subclassing :class:`~pfl.hyperparam.base.AlgorithmHyperParams`.
    As a rule of thumb, subclass algorithm parameters if this new algorithm
    should be suitable for subclassing as well.
    """

    def __init__(self):
        # The algorithm needs to keep a random state for any central random
        # operations.
        self._random_state = np.random.RandomState(
            np.random.randint(0, 2**32, dtype=np.uint32))
        self._current_central_iteration = 0

    def _get_seed(self):
        return self._random_state.randint(0, 2**32, dtype=np.uint32)

    def save(self, dir_path: str) -> None:
        state_path = os.path.join(dir_path, 'algorithm_checkpoint.json')
        with open(state_path, 'w') as f:
            json.dump(
                {'current_central_iteration': self._current_central_iteration},
                f)

    def load(self, dir_path: str) -> None:
        state_path = os.path.join(dir_path, 'algorithm_checkpoint.json')
        if not os.path.exists(state_path):
            raise CheckpointNotFoundError(state_path)
        with open(state_path) as f:
            state = json.load(f)
            # Resume the next iteration.
            self._current_central_iteration = state[
                'current_central_iteration'] + 1

    @abstractmethod
    def get_next_central_contexts(
        self,
        model: ModelType,
        iteration: int,
        algorithm_params: AlgorithmHyperParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[
            AlgorithmHyperParamsType, ModelHyperParamsType], ...]], ModelType,
               Metrics]:
        pass

    @abstractmethod
    def process_aggregated_statistics(
            self, central_context: CentralContext[AlgorithmHyperParamsType,
                                                  ModelHyperParamsType],
            aggregate_metrics: Metrics, model: ModelType,
            statistics: StatisticsType) -> Tuple[ModelType, Metrics]:
        """
        The server-side part of the computation.

        This should process aggregated model statistics and use them to update
        a model.
        If the algorithm performs multiple training central iterations at once,
        which can happen if the particular algorithm returns multiple contexts
        for training by `get_next_central_contexts`, then this method is
        called once for each central-context-statistics combination.

        :param central_context:
            Settings used to gather the metrics and statistics also given as
            input.
        :param aggregate_metrics:
            A :class:`~pfl.metrics.Metrics` object with
            aggregated metrics accumulated from local training on users.
        :param model:
            The model in its state before the aggregate statistics were
            processed.
        :param statistics:
            Aggregated model statistics from a cohort of devices (simulated or
            real).
        :returns:
            A metrics object with new metrics generated from this model update.
            Do not include any of the aggregate_metrics!
        """
        pass

    def process_aggregated_statistics_from_all_contexts(
            self, stats_context_pairs: Tuple[Tuple[
                CentralContext[AlgorithmHyperParamsType, ModelHyperParamsType],
                StatisticsType], ...], aggregate_metrics: Metrics,
            model: ModelType) -> Tuple[ModelType, Metrics]:
        """
        Override this method if the algorithm you are developing
        produces aggregated statistics from multiple cohorts using multiple
        central contexts each central iteration. Usually, algorithms only
        require one aggregated statistics from one cohort each central
        iteration, hence by default this method simply forwards that single
        result to ``process_aggregated_statistics``.

        :param stats_context_pairs:
            Tuple of pairs of central context and its model statistics.
            Each model statistics were accumulated from a cohort which used
            the corresponding central context.
        :param aggregate_metrics:
            A :class:`~pfl.metrics.Metrics` object with
            aggregated metrics accumulated from local training on users.
        :param model:
            The model in its state before the aggregate statistics were
            processed.
        :returns:
            A metrics object with new metrics generated from this model update.
            Do not include any of the aggregate_metrics!
        """
        if len(stats_context_pairs) != 1:
            raise ValueError(
                "The algorithm received model statistics from "
                "multiple cohorts. You need to override "
                "`process_aggregated_statistics_from_all_contexts` to handle "
                "the multiple results.")

        (central_context, statistics), = stats_context_pairs
        return self.process_aggregated_statistics(central_context,
                                                  aggregate_metrics, model,
                                                  statistics)

    def simulate_one_user(
        self, model: ModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[AlgorithmHyperParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        Simulate the client-side part of the computation.

        This should train a single user on its dataset (simulated as
        ``user_dataset``) and return the model statistics.
        The statistics should be in such a form that they can be aggregated
        over users.
        For a cohort of multiple users, this method will be called multiple
        times.

        :param model:
            The current state of the model.
        :param user_dataset:
            The data simulated for a single user.
        :param central_context:
            Settings to use for this round. Contains the model params
            used for training/evaluating a user's model.
        :returns:
            A tuple ``(statistics, metrics)``, with model statistics and metrics
            generated from the training of a single user.
            Both statistics and metrics will be aggregated and the aggregate
            will be passed to ``process_aggregated_statistics``.
        """
        raise NotImplementedError

    def run(self,
            algorithm_params: AlgorithmHyperParamsType,
            backend: Backend,
            model: ModelType,
            model_train_params: ModelHyperParamsType,
            model_eval_params: Optional[ModelHyperParamsType] = None,
            callbacks: Optional[List[TrainingProcessCallback]] = None,
            *,
            send_metrics_to_platform: bool = True) -> ModelType:
        """
        Orchestrate the federated computation.

        :param backend:
            The :class:`~pfl.aggregate.base.Backend` that aggregates the
            contributions from individual users.
            This may be simulated (in which case the backend will call
            ``simulate_one_user``), or it may perform for live training (in
            which case your client code will be called).
        :param model:
            The model to train.
        :param callbacks:
            A list of callbacks for hooking into the training loop, potentially
            performing complementary actions to the model training, e.g. central
            evaluation or training parameter schemes.
        :param send_metrics_to_platform:
            Allow the platform to process the aggregated metrics after
            each central iteration.
        :returns:
            The trained model.
            It may be the same object as given in the input, or it may be
            different.

        """
        self._current_central_iteration = 0
        has_reported_on_train_metrics = False
        should_stop = False
        callbacks = list(callbacks or [])
        default_callbacks = get_platform().get_default_callbacks()
        for default_callback in default_callbacks:
            # Add default callback if it is not in the provided callbacks
            if all(
                    type(callback) != type(default_callback)
                    for callback in callbacks):
                logger.debug(f'Adding {default_callback}')
                callbacks.append(default_callback)
            else:
                logger.debug(f'Not adding duplicate {default_callback}')

        on_train_metrics = Metrics()
        for callback in callbacks:
            on_train_metrics |= callback.on_train_begin(model=model)

        central_contexts = None
        while True:
            # Step 1
            # Get instructions from algorithm what to run next.
            # Can be multiple queries to cohorts of devices.
            (new_central_contexts, model,
             all_metrics) = self.get_next_central_contexts(
                 model, self._current_central_iteration, algorithm_params,
                 model_train_params, model_eval_params)
            if new_central_contexts is None:
                break
            else:
                central_contexts = new_central_contexts

            if not has_reported_on_train_metrics:
                all_metrics |= on_train_metrics
                has_reported_on_train_metrics = True

            # Step 2
            # Get aggregated model updates and
            # metrics from the requested queries.
            results: List[Tuple[StatisticsType,
                                Metrics]] = algorithm_utils.run_train_eval(
                                    self, backend, model, central_contexts)

            # Step 3
            # For each query result, accumulate metrics and
            # let model handle statistics result if query had any.
            stats_context_pairs = []
            for central_context, (stats,
                                  metrics) in zip(central_contexts, results):
                all_metrics |= metrics
                if stats is not None:
                    stats_context_pairs.append((central_context, stats))
            # Process statistics and get new model.
            (model, update_metrics
             ) = self.process_aggregated_statistics_from_all_contexts(
                 tuple(stats_context_pairs), all_metrics, model)

            all_metrics |= update_metrics

            # Step 4
            # End-of-iteration callbacks
            for callback in callbacks:
                stop_signal, callback_metrics = (
                    callback.after_central_iteration(
                        all_metrics,
                        model,
                        central_iteration=self._current_central_iteration))
                all_metrics |= callback_metrics
                should_stop |= stop_signal

            if send_metrics_to_platform:
                get_platform().consume_metrics(
                    all_metrics, iteration=self._current_central_iteration)

            if should_stop:
                break
            self._current_central_iteration += 1

        for callback in callbacks:
            # Calls with central iteration configs used for final round.
            callback.on_train_end(model=model)

        return model


@dataclass(frozen=True)
class NNAlgorithmParams(AlgorithmHyperParams):
    """
    Parameters for algorithms that involve training
    neural networks.

    :param central_num_iterations:
        Total number of central iterations.
    :param evaluation_frequency:
        Frequency with which the model will be evaluated (in terms
        of central iterations).
    :param train_cohort_size:
        Cohort size for training.
    :param val_cohort_size:
        Cohort size for evaluation on validation users.
    """
    central_num_iterations: int
    evaluation_frequency: int
    train_cohort_size: HyperParamClsOrInt
    val_cohort_size: Optional[int]


NNAlgorithmParamsType = TypeVar('NNAlgorithmParamsType',
                                bound=NNAlgorithmParams)


class FederatedNNAlgorithm(FederatedAlgorithm[NNAlgorithmParamsType,
                                              ModelHyperParamsType,
                                              StatefulModelType,
                                              StatisticsType,
                                              AbstractDatasetType]):

    def __init__(self):
        super().__init__()
        # Just a placeholder of tensors to get_parameters faster.
        self._initial_model_state = None

    @abstractmethod
    def train_one_user(
        self, initial_model_state: StatisticsType, model: StatefulModelType,
        user_dataset: AbstractDatasetType,
        central_context: CentralContext[NNAlgorithmParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[StatisticsType, Metrics]:
        pass

    def get_next_central_contexts(
        self,
        model: StatefulModelType,
        iteration: int,
        algorithm_params: NNAlgorithmParamsType,
        model_train_params: ModelHyperParamsType,
        model_eval_params: Optional[ModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[CentralContext[NNAlgorithmParamsType,
                                             ModelHyperParamsType], ...]],
               StatefulModelType, Metrics]:
        if iteration == 0:
            self._initial_model_state = None

        # Stop condition for iterative NN federated algorithms.
        if iteration == algorithm_params.central_num_iterations:
            return None, model, Metrics()

        do_evaluation = iteration % algorithm_params.evaluation_frequency == 0
        static_model_train_params: ModelHyperParamsType = \
            model_train_params.static_clone()
        static_model_eval_params: Optional[ModelHyperParamsType]
        static_model_eval_params = None if model_eval_params is None else model_eval_params.static_clone(
        )

        configs: List[CentralContext[
            NNAlgorithmParamsType, ModelHyperParamsType]] = [
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=get_param_value(
                        algorithm_params.train_cohort_size),
                    population=Population.TRAIN,
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    algorithm_params=algorithm_params.static_clone(),
                    seed=self._get_seed())
            ]
        if do_evaluation and algorithm_params.val_cohort_size:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=algorithm_params.val_cohort_size,
                    population=Population.VAL,
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    algorithm_params=algorithm_params.static_clone(),
                    seed=self._get_seed()))

        return tuple(configs), model, Metrics()

    def simulate_one_user(
        self, model: StatefulModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[NNAlgorithmParamsType,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        If population is ``Population.TRAIN``, trains one user and returns the
        model difference before and after training.
        Also evaluates the performance before and after training the user.
        Metrics with the postfix "after local training" measure the performance
        after training the user.
        If population is not ``Population.TRAIN``, does only evaluation.
        """
        # pytype: disable=duplicate-keyword-argument
        initial_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=False)
        final_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=True)
        # pytype: enable=duplicate-keyword-argument

        metrics = Metrics()
        # Train local user.

        if central_context.population == Population.TRAIN:

            if central_context.do_evaluation:
                metrics |= model.evaluate(user_dataset,
                                          initial_metrics_format_fn,
                                          central_context.model_eval_params)

            self._initial_model_state = model.get_parameters(
                self._initial_model_state)
            statistics, train_metrics = self.train_one_user(
                self._initial_model_state, model, user_dataset,
                central_context)
            metrics |= train_metrics

            # Evaluate after local training.
            if central_context.do_evaluation:
                metrics |= model.evaluate(user_dataset,
                                          final_metrics_format_fn,
                                          central_context.model_eval_params)

            model.set_parameters(self._initial_model_state)
            return statistics, metrics
        else:
            metrics = model.evaluate(user_dataset, initial_metrics_format_fn,
                                     central_context.model_eval_params)
            return None, metrics


@dataclass(frozen=True)
class PersonalizedNNAlgorithmParams(NNAlgorithmParams):
    """
    Base parameter config for federated personalization
    algorithms.

    Has same parameters as
    :class:`~pfl.algorithm.base.NNAlgorithmParams` in addition to:

    :param val_split_fraction:
        Parameter to ``Dataset.split``. Defines the fraction of
        data to use for training. A good starting value can be
        ``0.8`` in many cases, i.e. split into 80% training data
        and 20% validation data.
    :param min_train_size:
        Parameter to ``Dataset.split``. Defines the minimum
        number of data samples for training.
        A good starting value can be ``1`` in many cases.
    :param min_val_size:
        Parameter to ``Dataset.split``. Defines the minimum
        number of data samples for validation.
        A good starting value can be ``1`` in many cases.
    """
    val_split_fraction: float
    min_train_size: int
    min_val_size: int


class PersonalizedNNAlgorithm(
        FederatedNNAlgorithm[PersonalizedNNAlgorithmParams,
                             ModelHyperParamsType, StatefulModelType,
                             StatisticsType, AbstractDatasetType]):

    def simulate_one_user(
        self, model: StatefulModelType, user_dataset: AbstractDatasetType,
        central_context: CentralContext[PersonalizedNNAlgorithmParams,
                                        ModelHyperParamsType]
    ) -> Tuple[Optional[StatisticsType], Metrics]:
        """
        If trains one user and returns the model difference before and after
        training. Also evaluates the performance before and after training the
        user. Metrics with the postfix "after local training" measure the
        performance after training the user. Unlike in FederatedNNAlgorithm,
        the training happens whether the user is in the train population or
        not.

        The performance after training on the val population is a measurement
        of how well the model can personalize to one user.
        """
        algo_params = central_context.algorithm_params
        train_dataset, val_dataset = user_dataset.split(
            algo_params.val_split_fraction, algo_params.min_train_size,
            algo_params.min_val_size)

        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        initial_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=False)
        final_metrics_format_fn = lambda n: TrainMetricName(
            n, central_context.population, after_training=True)
        initial_val_metrics_format_fn = lambda n: TrainMetricName(
            n,
            central_context.population,
            after_training=False,
            local_partition='val')
        final_val_metrics_format_fn = lambda n: TrainMetricName(
            n,
            central_context.population,
            after_training=True,
            local_partition='val')
        # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        metrics = Metrics()

        if central_context.do_evaluation:
            # Evaluate on train and eval partition before training.
            metrics |= model.evaluate(train_dataset, initial_metrics_format_fn,
                                      central_context.model_eval_params)
            metrics |= model.evaluate(val_dataset,
                                      initial_val_metrics_format_fn,
                                      central_context.model_eval_params)

        self._initial_model_state = model.get_parameters(
            self._initial_model_state)
        statistics, train_metrics = self.train_one_user(
            self._initial_model_state, model, train_dataset, central_context)
        metrics |= train_metrics

        # Evaluate after local training.
        if central_context.do_evaluation:
            metrics |= model.evaluate(train_dataset, final_metrics_format_fn,
                                      central_context.model_eval_params)
            metrics |= model.evaluate(val_dataset, final_val_metrics_format_fn,
                                      central_context.model_eval_params)

        model.set_parameters(self._initial_model_state)
        if central_context.population != Population.TRAIN:
            # Only report model update to aggregate for train population.
            return None, metrics
        else:
            return statistics, metrics
