# Copyright © 2023-2024 Apple Inc.
import argparse
import logging
import os
from functools import partial
from uuid import uuid4

import mlx
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_mlx
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_dnn_training_arguments,
    add_filepath_arguments,
    add_mechanism_arguments,
    add_seed_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
)
from utils.logging import init_logging

from pfl.aggregate.simulate import SimulatedBackend
from pfl.aggregate.weighting import WeightByDatapoints, WeightByUser
from pfl.callback import (
    AggregateMetricsToDisk,
    CentralEvaluationCallback,
    ModelCheckpointingCallback,
    StopwatchCallback,
    TrackBestOverallMetrics,
    WandbCallback,
)
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.mlx_ops import to_tensor
from pfl.model.mlx import MLXModel
from pfl.privacy import CentrallyAppliedPrivacyMechanism


def main():
    init_logging(logging.DEBUG)
    maybe_inject_arguments_from_config()
    logger = logging.getLogger(name=__name__)

    argument_parser = argparse.ArgumentParser(
        description=
        'Train a model using private federated learning in simulation.')

    argument_parser = add_dnn_training_arguments(argument_parser)
    argument_parser = add_mechanism_arguments(argument_parser)
    argument_parser = add_filepath_arguments(argument_parser)
    argument_parser = add_seed_arguments(argument_parser)
    argument_parser = add_algorithm_arguments(argument_parser)
    argument_parser = add_dataset_arguments(argument_parser)
    argument_parser = add_model_arguments(argument_parser)
    arguments = argument_parser.parse_args()

    mx.random.seed(arguments.seed)
    np.random.seed(arguments.seed)

    local_privacy = parse_mechanism(
        mechanism_name=arguments.local_privacy_mechanism,
        clipping_bound=arguments.local_privacy_clipping_bound,
        epsilon=arguments.local_epsilon,
        delta=arguments.local_delta,
        order=arguments.local_order)

    central_privacy = parse_mechanism(
        mechanism_name=arguments.central_privacy_mechanism,
        clipping_bound=arguments.central_privacy_clipping_bound,
        epsilon=arguments.central_epsilon,
        delta=arguments.central_delta,
        order=arguments.central_order,
        cohort_size=arguments.cohort_size,
        noise_cohort_size=arguments.noise_cohort_size,
        num_epochs=arguments.central_num_iterations,
        min_separation=arguments.min_separation,
        population=1e6)

    arguments.numpy_to_tensor = partial(to_tensor, dtype=None)
    (training_federated_dataset, val_federated_dataset, central_data,
     _) = get_datasets(arguments)

    mlx_model = get_model_mlx(arguments)
    model = MLXModel(model=mlx_model,
                     local_optimizer=mlx.optimizers.SGD(0.0),
                     central_optimizer=mlx.optimizers.SGD(
                         learning_rate=arguments.learning_rate))

    weighting_strategy = WeightByDatapoints(
    ) if arguments.weight_by_samples else WeightByUser()

    postprocessors = [
        weighting_strategy, local_privacy,
        CentrallyAppliedPrivacyMechanism(central_privacy)
    ]

    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=postprocessors)

    algorithm, algorithm_params, algorithm_callbacks = get_algorithm(arguments)

    model_train_params = NNTrainHyperParams(
        local_learning_rate=arguments.local_learning_rate,
        local_num_epochs=arguments.local_num_epochs,
        local_batch_size=arguments.local_batch_size
        if arguments.local_batch_size > 0 else None)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=arguments.central_eval_batch_size)

    # Central evaluation on val data.
    central_evaluation_cb = CentralEvaluationCallback(
        central_data,
        model_eval_params=model_eval_params,
        frequency=arguments.evaluation_frequency)

    if arguments.restore_model_path is not None:
        model.load(arguments.restore_model_path)
        logger.info(f'Restored model from {arguments.restore_model_path}')

    callbacks = [
        StopwatchCallback(),
        central_evaluation_cb,
        AggregateMetricsToDisk('./metrics.csv'),
        TrackBestOverallMetrics(
            higher_is_better_metric_names=['Central val | accuracy']),
    ]
    callbacks.extend(algorithm_callbacks)

    if arguments.save_model_path is not None:
        callbacks.append(ModelCheckpointingCallback(arguments.save_model_path))

    if arguments.wandb_project_id:
        callbacks.append(
            WandbCallback(
                wandb_project_id=arguments.wandb_project_id,
                wandb_experiment_name=os.environ.get('WANDB_TASK_ID',
                                                     str(uuid4())),
                # List of dicts to one dict.
                wandb_config=dict(vars(arguments)),
                tags=os.environ.get('WANDB_TAGS', 'empty-tag').split(','),
                group=os.environ.get('WANDB_GROUP', None)))

    model = algorithm.run(algorithm_params=algorithm_params,
                          backend=backend,
                          model=model,
                          model_train_params=model_train_params,
                          model_eval_params=model_eval_params,
                          callbacks=callbacks)


if __name__ == '__main__':
    main()
