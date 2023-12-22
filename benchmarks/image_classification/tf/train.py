# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import argparse
import logging

import numpy as np
import tensorflow as tf  # type: ignore

from pfl.algorithm import (FederatedAveraging, NNAlgorithmParams)
from pfl.aggregate.simulate import SimulatedBackend
from pfl.aggregate.weighting import WeightByDatapoints, WeightByUser
from pfl.callback import (AggregateMetricsToDisk, CentralEvaluationCallback,
                          StopwatchCallback, ModelCheckpointingCallback)
from pfl.hyperparam import NNTrainHyperParams, NNEvalHyperParams
from pfl.model.tensorflow import TFModel
from pfl.privacy import CentrallyAppliedPrivacyMechanism

from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_tf2
from utils.argument_parsing import (
    add_algorithm_arguments, add_filepath_arguments, add_seed_arguments,
    add_dnn_training_arguments, add_mechanism_arguments, get_algorithm,
    parse_mechanism, maybe_inject_arguments_from_config)
from utils.logging import init_logging


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

    tf.random.set_seed(arguments.seed)
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

    (training_federated_dataset, val_federated_dataset, central_data,
     _) = get_datasets(arguments)
    keras_model = get_model_tf2(arguments)
    keras_model.compile(
        # Learning rate will be overridden by `local_learning_rate` in simulation.
        tf.keras.optimizers.SGD(0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    print(keras_model.summary())

    metrics = {
        'loss':
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy()
    }

    central_optimizer = tf.keras.optimizers.SGD(arguments.learning_rate)
    model = TFModel(
        model=keras_model,
        central_optimizer=central_optimizer,
        metrics=metrics)

    weighting_strategy = WeightByDatapoints(
    ) if arguments.weight_by_samples else WeightByUser()

    postprocessors = [
        weighting_strategy, local_privacy,
        CentrallyAppliedPrivacyMechanism(central_privacy)
    ]

    backend = SimulatedBackend(
        training_data=training_federated_dataset,
        val_data=val_federated_dataset,
        postprocessors=postprocessors)

    algorithm, algorithm_params = get_algorithm(arguments)

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
        logger.info('Restored model from {}'.format(
            arguments.restore_model_path))

    model = algorithm.run(
        algorithm_params=algorithm_params,
        backend=backend,
        model=model,
        model_train_params=model_train_params,
        model_eval_params=model_eval_params,
        callbacks=[
            StopwatchCallback(),
            central_evaluation_cb,
            ModelCheckpointingCallback('./checkpoints'),
            AggregateMetricsToDisk('./metrics.csv'),
        ])


if __name__ == '__main__':
    main()
