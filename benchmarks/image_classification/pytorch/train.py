# Copyright © 2023-2024 Apple Inc.
import argparse
import logging
import os

import numpy as np
import torch  # type: ignore
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
from pfl.internal.ops.pytorch_ops import to_tensor
from pfl.model.pytorch import PyTorchModel
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

    torch.random.manual_seed(arguments.seed)
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

    arguments.numpy_to_tensor = to_tensor
    (training_federated_dataset, val_federated_dataset, central_data,
     _) = get_datasets(arguments)

    pytorch_model = get_model_pytorch(arguments)

    params = [p for p in pytorch_model.parameters() if p.requires_grad]

    model = PyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=torch.optim.SGD(
                             params, arguments.learning_rate))

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
        # Uncomment to save central model checkpoints during training.
        #ModelCheckpointingCallback('./checkpoints'),
        AggregateMetricsToDisk('./metrics.csv'),
        TrackBestOverallMetrics(
            higher_is_better_metric_names=['Central val | accuracy']),
    ]
    callbacks.extend(algorithm_callbacks)

    if arguments.wandb_project_id:
        assert 'TASK_ID' in os.environ, "Wandb needs a task id"
        callbacks.append(
            WandbCallback(
                wandb_project_id=arguments.wandb_project_id,
                wandb_experiment_name=os.environ['TASK_ID'],
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
