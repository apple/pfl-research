# Copyright © 2023-2024 Apple Inc.
import argparse
import logging
import os

import numpy as np
import tensorflow as tf  # type: ignore
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.argument_parsing import add_model_arguments, get_model_tf2
from model.tf2.metrics import MaskedCategoricalAccuracy, MaskedCategoricalCrossentropy, Perplexity
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
from utils.callback.tensorflow import CentralLRDecay, LocalLRDecay
from utils.logging import init_logging

from pfl.aggregate.simulate import SimulatedBackend
from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import (
    AggregateMetricsToDisk,
    CentralEvaluationCallback,
    ModelCheckpointingCallback,
    StopwatchCallback,
    TensorBoardCallback,
)
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.platform import get_platform
from pfl.model.tensorflow import TFModel
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
    tf.random.set_seed(arguments.seed)

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
        population=arguments.population)

    # create federated training and val datasets and a central val dataset.
    arguments.include_mask = False
    (training_federated_dataset, val_federated_dataset, central_data,
     metadata) = get_datasets(arguments)
    PAD = metadata['pad_symbol']
    UNK = metadata['unk_symbol']
    arguments.vocab_size = metadata['vocab_size']
    arguments.max_sequence_length = metadata['max_sequence_length']

    keras_model = get_model_tf2(arguments)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    keras_model.compile(tf.keras.optimizers.SGD(0.0), loss=loss_function)

    if arguments.central_optimizer == 'adam':
        # Hyperparameters for stability, see S. Reddi et al. 2020 Appendix C.1.
        central_optimizer = tf.keras.optimizers.Adam(
            arguments.learning_rate,
            epsilon=arguments.adaptivity_degree,
            beta_2=0.99)
    else:
        assert arguments.central_optimizer == 'sgd'
        central_optimizer = tf.keras.optimizers.SGD(arguments.learning_rate)

    metrics = {
        'loss':
        MaskedCategoricalCrossentropy(from_logits=True,
                                      masked_tokens=[PAD, UNK]),
        'next word accuracy | including unk':
        MaskedCategoricalAccuracy(masked_tokens=[PAD]),
        'next word accuracy':
        MaskedCategoricalAccuracy(masked_tokens=[PAD, UNK]),
        'perplexity':
        Perplexity(from_logits=True, masked_tokens=[PAD, UNK])
    }

    model = TFModel(model=keras_model,
                    central_optimizer=central_optimizer,
                    metrics=metrics)

    weighting_strategy = parse_weighting_strategy(arguments.weighting,
                                                  arguments.weight_clip)

    postprocessors = [
        weighting_strategy, local_privacy,
        CentrallyAppliedPrivacyMechanism(central_privacy)
    ]

    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=postprocessors)

    algorithm, algorithm_params = get_algorithm(arguments)

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
        ModelCheckpointingCallback('./checkpoints'),
        AggregateMetricsToDisk('./metrics.csv'),
    ]
    if arguments.central_lr_num_warmup_iterations > 0:
        central_lr_warmup_cb = CentralLRDecay(
            arguments.learning_rate,
            arguments.learning_rate,
            arguments.central_num_iterations,
            arguments.central_lr_num_warmup_iterations,
            linear_warmup=True)
        callbacks.append(central_lr_warmup_cb)
    if arguments.local_lr_decay:
        raise NotImplementedError(
            "TODO: rdar://109165296 Implement LocalLRDecay "
            "as an adaptive hyperparameter")
    if arguments.fedsgd_after_amount_trained is not None:
        raise NotImplementedError(
            "TODO: rdar://109165050 Implement DecayToFedSGD "
            "as an adaptive hyperparameter")

    if arguments.restore_model_path is not None:
        model.load(arguments.restore_model_path)
        logger.info(f'Restored model from {arguments.restore_model_path}')

    if arguments.use_tensorboard:
        tb_port = os.environ.get('TENSORBOARD_PORT', None)
        tb_dir, = get_platform().create_checkpoint_directories(['logs'])
        callbacks.append(
            TensorBoardCallback(
                tb_dir,
                # Write weights and graph can be useful for debugging purposes.
                write_weights=False,
                write_graph=False,
                tensorboard_port=int(tb_port) if tb_port else None))

    model = algorithm.run(algorithm_params=algorithm_params,
                          backend=backend,
                          model=model,
                          model_train_params=model_train_params,
                          model_eval_params=model_eval_params,
                          callbacks=callbacks)


if __name__ == '__main__':
    main()
