# Copyright © 2023-2024 Apple Inc.

import argparse
import logging
import os
from functools import partial
from uuid import uuid4

import numpy as np
import torch
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_pytorch
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
from utils.callback.pytorch import get_polynomial_decay_schedule_with_warmup
from utils.logging import init_logging

from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import (
    AggregateMetricsToDisk,
    CentralEvaluationCallback,
    ModelCheckpointingCallback,
    StopwatchCallback,
    TrackBestOverallMetrics,
    WandbCallback,
)
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.pytorch_ops import to_tensor
from pfl.model.pytorch import PyTorchModel

from .argument_parsing import add_flair_training_arguments


def main():
    init_logging(logging.DEBUG)
    maybe_inject_arguments_from_config()
    logger = logging.getLogger(name=__name__)

    argument_parser = argparse.ArgumentParser(
        description=
        'Train a model using private federated learning in simulation.')

    argument_parser = add_dataset_arguments(argument_parser)
    argument_parser = add_model_arguments(argument_parser)

    argument_parser = add_filepath_arguments(argument_parser)
    argument_parser = add_seed_arguments(argument_parser)
    argument_parser = add_algorithm_arguments(argument_parser)
    argument_parser = add_flair_training_arguments(argument_parser)
    argument_parser = add_dnn_training_arguments(argument_parser)
    argument_parser = add_mechanism_arguments(argument_parser)
    arguments = argument_parser.parse_args()

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

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
        population=arguments.population,
        min_separation=arguments.min_separation,
        is_central=True)

    # to_tensor is float32 by default,
    # training faster if input images remain in uint8.
    arguments.numpy_to_tensor = partial(to_tensor, dtype=None)
    (training_federated_dataset, val_federated_dataset, central_data,
     metadata) = get_datasets(arguments)

    num_classes = len(metadata["label_mapping"])
    arguments.channel_mean = metadata["channel_mean"]
    arguments.channel_stddevs = metadata["channel_stddevs"]
    arguments.num_classes = num_classes

    pytorch_model = get_model_pytorch(arguments)

    variables = [p for p in pytorch_model.parameters() if p.requires_grad]
    if arguments.central_optimizer == 'adam':
        central_optimizer = torch.optim.AdamW(
            variables,
            arguments.learning_rate,
            eps=0.01,
            betas=(0.9, 0.99),
            weight_decay=arguments.weight_decay)
    else:
        central_optimizer = torch.optim.SGD(
            variables,
            arguments.learning_rate,
            weight_decay=arguments.weight_decay)

    central_lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        central_optimizer,
        num_warmup_steps=30,
        num_training_steps=arguments.central_num_iterations,
        lr_end=0.02)

    model = PyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=central_optimizer,
                         central_learning_rate_scheduler=central_lr_scheduler)

    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=[local_privacy, central_privacy])

    algorithm, algorithm_params, algorithm_callbacks = get_algorithm(arguments)

    model_train_params = NNTrainHyperParams(
        local_learning_rate=arguments.local_learning_rate,
        local_num_epochs=arguments.local_num_epochs,
        local_batch_size=arguments.local_batch_size,
        local_max_grad_norm=10.0)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=arguments.central_eval_batch_size)

    # Central evaluation on dev data.
    callbacks = [
        CentralEvaluationCallback(central_data,
                                  model_eval_params=model_eval_params,
                                  frequency=arguments.evaluation_frequency),
        StopwatchCallback(),
        AggregateMetricsToDisk('./metrics.csv'),
        TrackBestOverallMetrics(
            higher_is_better_metric_names=['Central val | macro AP']),
    ]

    if arguments.restore_model_path is not None:
        model.load(arguments.restore_model_path)
        logger.info(f'Restored model from {arguments.restore_model_path}')

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
