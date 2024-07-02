# Copyright © 2023-2024 Apple Inc.
import argparse
import logging
import os
from uuid import uuid4

import numpy as np
import mlx
import mlx.core as mx
import mlx.nn as nn
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_mlx
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_filepath_arguments,
    add_seed_arguments,
    add_weighting_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
    parse_weighting_strategy,
)
from utils.callback.mlx import get_polynomial_decay_schedule_with_warmup
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
from pfl.model.mlx import MLXModel
from pfl.privacy import CentrallyAppliedPrivacyMechanism

from ..argument_parsing import add_lm_arguments


def main():
    init_logging(logging.DEBUG)
    maybe_inject_arguments_from_config()
    logger = logging.getLogger(name=__name__)

    parser = argparse.ArgumentParser(
        description=
        'Train a model using private federated learning in simulation.')

    parser = add_filepath_arguments(parser)
    parser = add_seed_arguments(parser)
    parser = add_algorithm_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_model_arguments(parser)
    parser = add_lm_arguments(parser)
    parser = add_weighting_arguments(parser)
    arguments = parser.parse_args()

    np.random.seed(arguments.seed)
    mx.random.seed(arguments.seed)

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
        min_separation=arguments.min_separation)

    # create federated training and val datasets and a central val dataset.
    arguments.include_mask = False
    (training_federated_dataset, val_federated_dataset, central_data,
     metadata) = get_datasets(arguments)

    arguments.vocab_size = metadata['vocab_size']
    arguments.pad_symbol = metadata['pad_symbol']
    arguments.unk_symbol = metadata['unk_symbol']
    arguments.max_sequence_length = metadata['max_sequence_length']

    mlx_model = get_model_mlx(arguments)

    if arguments.central_lr_num_warmup_iterations > 0:
        central_learning_rate = get_polynomial_decay_schedule_with_warmup(
                lr_init=arguments.learning_rate,
                num_warmup_steps=arguments.central_lr_num_warmup_iterations,
                num_training_steps=arguments.central_num_iterations,
                                                    lr_end=arguments.learning_rate)
    else:
        central_learning_rate = arguments.learning_rate

    if arguments.central_optimizer == 'adam':
        # Hyperparameters for stability, see S. Reddi et al. 2020 Appendix C.1.
        central_optimizer = mlx.optimizers.Adam(central_learning_rate,
                                             eps=arguments.adaptivity_degree,
                                             betas=(0.9, 0.99))
    else:
        assert arguments.central_optimizer == 'sgd'
        central_optimizer = mlx.optimizers.SGD(central_learning_rate)

    model = MLXModel(model=mlx_model,
                         local_optimizer=mlx.optimizers.SGD(0.0),
                         central_optimizer=central_optimizer)

    weighting_strategy = parse_weighting_strategy(arguments.weighting,
                                                  arguments.weight_clip)

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
        local_batch_size=arguments.local_batch_size)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=arguments.central_eval_batch_size)

    callbacks = [
        StopwatchCallback(),
        CentralEvaluationCallback(central_data,
                                  model_eval_params=model_eval_params,
                                  frequency=arguments.evaluation_frequency),
        AggregateMetricsToDisk('./metrics.csv'),
        TrackBestOverallMetrics(
            lower_is_better_metric_names=['Central val | perplexity']),
    ]
    if arguments.fedsgd_after_amount_trained is not None:
        raise NotImplementedError(
            "TODO: rdar://109165050 Implement DecayToFedSGD "
            "as an adaptive hyperparameter")

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
