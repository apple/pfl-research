import argparse
import logging

import numpy as np
import torch
import transformers
from dataset.argument_parsing import add_dataset_arguments, get_datasets
from model.hugging_face import causal_lm_metrics_fn, wrap_hugging_face_model
from utils.argument_parsing import (
    add_algorithm_arguments,
    add_filepath_arguments,
    add_seed_arguments,
    get_algorithm,
    maybe_inject_arguments_from_config,
    parse_mechanism,
)
from utils.logging import init_logging

from llm.argument_parsing import add_llm_arguments, parse_peft_config
from pfl.aggregate.simulate import SimulatedBackend
from pfl.callback import AggregateMetricsToDisk, CentralEvaluationCallback, StopwatchCallback
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel


def main():
    init_logging(logging.INFO)
    maybe_inject_arguments_from_config()
    logger = logging.getLogger(name=__name__)

    parser = argparse.ArgumentParser(
        description=
        'Train a model using private federated learning in simulation.')

    parser = add_algorithm_arguments(parser)
    parser = add_filepath_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_seed_arguments(parser)
    parser = add_llm_arguments(parser)
    arguments = parser.parse_args()

    np.random.seed(arguments.seed)
    torch.random.manual_seed(arguments.seed)
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        arguments.hugging_face_model_name_or_path,
        model_max_length=arguments.model_max_length,
        padding_side=arguments.padding_side,
        use_fast=arguments.use_fast_tokenizer)
    arguments.tokenizer = tokenizer

    logger.info("Loading dataset.")
    training_data, val_data, central_data, metadata = get_datasets(arguments)

    logger.info(f"Loading model from "
                f"{arguments.hugging_face_model_name_or_path}.")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        arguments.hugging_face_model_name_or_path)
    if "postprocessing_model_fn" in metadata:
        # Additional post-processing with loaded model
        metadata["postprocessing_model_fn"](hf_model)

    # Parameter efficient fine-tuning
    peft_config = parse_peft_config(arguments)
    hf_model = wrap_hugging_face_model(hf_model, peft_config,
                                       causal_lm_metrics_fn)

    params = [p for p in hf_model.parameters() if p.requires_grad]
    if arguments.central_optimizer == 'adam':
        # Hyperparameters for stability, see S. Reddi et al. 2020 Appendix C.1.
        central_optimizer = torch.optim.Adam(params,
                                             arguments.learning_rate,
                                             eps=arguments.adaptivity_degree,
                                             betas=(0.9, 0.99))
    else:
        assert arguments.central_optimizer == 'sgd'
        central_optimizer = torch.optim.SGD(params, arguments.learning_rate)

    model = PyTorchModel(
        model=hf_model,
        local_optimizer_create=torch.optim.SGD,
        central_optimizer=central_optimizer,
        amp_dtype=getattr(torch, arguments.amp_dtype),
        grad_scaling=arguments.grad_scaling,
        model_dtype_same_as_amp=arguments.model_dtype_same_as_amp)

    backend = SimulatedBackend(training_data=training_data,
                               val_data=val_data,
                               postprocessors=[local_privacy, central_privacy])

    model_train_params = NNTrainHyperParams(
        local_learning_rate=arguments.local_learning_rate,
        local_num_epochs=arguments.local_num_epochs,
        local_batch_size=arguments.local_batch_size,
        local_max_grad_norm=arguments.local_max_grad_norm,
        grad_accumulation_steps=arguments.grad_accumulation_steps)

    model_eval_params = NNEvalHyperParams(
        local_batch_size=arguments.central_eval_batch_size)

    callbacks = [
        StopwatchCallback(),
        CentralEvaluationCallback(central_data,
                                  model_eval_params=model_eval_params,
                                  frequency=arguments.evaluation_frequency),
        AggregateMetricsToDisk('./metrics.csv')
    ]
    algorithm, algorithm_params, algorithm_callbacks = get_algorithm(arguments)
    callbacks.extend(algorithm_callbacks)

    logger.info("Starts federated learning.")
    model = algorithm.run(algorithm_params=algorithm_params,
                          backend=backend,
                          model=model,
                          model_train_params=model_train_params,
                          model_eval_params=model_eval_params,
                          callbacks=callbacks)


if __name__ == '__main__':
    main()
