# Copyright © 2024 Apple Inc.
import functools
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import app, flags, logging
from compare_utils.tensorflow import simple_cnn
from utils import utils_impl
from utils.datasets import cifar10_dataset
from utils.optimizers import optimizer_utils

CIFAR_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

with utils_impl.record_hparam_flags() as optimizer_flags:
    # Defining optimizer flags
    optimizer_utils.define_optimizer_flags("client")
    optimizer_utils.define_optimizer_flags("server")
    optimizer_utils.define_lr_schedule_flags("client")
    optimizer_utils.define_lr_schedule_flags("server")

with utils_impl.record_hparam_flags() as shared_flags:
    # Federated training hyperparameters
    flags.DEFINE_integer(
        "local_num_epochs",
        1,
        "Number of epochs in the client to take per round.",
    )
    flags.DEFINE_integer("local_batch_size", 10, "Batch size on the clients.")
    flags.DEFINE_integer("cohort_size", 50,
                         "How many clients to sample per round.")
    flags.DEFINE_integer("seed", 1, "Random seed")

    # Training loop configuration
    flags.DEFINE_integer("central_num_iterations", 1500,
                         "Number of total training rounds.")
    flags.DEFINE_integer(
        "evaluation_frequency",
        1,
        "How often to evaluate the global model on the validation dataset.",
    )

FLAGS = flags.FLAGS


def main(argv):
    start_time_total = time.perf_counter()
    if len(argv) > 1:
        raise app.UsageError("Expected no command-line arguments, "
                             f"got: {argv}")

    tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    """
    # Attempted to share GPU for multiple processes, but no difference.
    #From https://www.tensorflow.org/federated/tutorials/simulations_with_accelerators
    for gpu_device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu_device, True)
        tf.config.set_logical_device_configuration(
        gpu_device,
        [tf.config.LogicalDeviceConfiguration(memory_limit=10240) for _ in range(4)])
    """

    gpu_devices = tf.config.list_logical_devices("GPU")
    print("logical devices", gpu_devices)
    if len(gpu_devices) > 0:
        tff.backends.native.set_local_python_execution_context(
            default_num_clients=FLAGS.cohort_size,
            max_fanout=2 * FLAGS.cohort_size,
            server_tf_device=tf.config.list_logical_devices("CPU")[0],
            client_tf_devices=gpu_devices,
            clients_per_thread=50,
        )

    client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
        "client")
    server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
        "server")

    def iterative_process_builder(
        model_fn, ) -> tff.templates.IterativeProcess:
        return tff.learning.algorithms.build_unweighted_fed_avg(
            model_fn=model_fn,
            client_optimizer_fn=client_optimizer_fn,
            server_optimizer_fn=server_optimizer_fn,
            use_experimental_simulation_loop=True,
        )

    crop_shape = (32, 32, 3)

    cifar_train, _ = cifar10_dataset.load_cifar10_federated(num_clients=1000)

    train_preprocess_fn = cifar10_dataset.create_preprocess_fn(
        num_epochs=FLAGS.local_num_epochs,
        batch_size=FLAGS.local_batch_size,
        shuffle_buffer_size=1000,
        crop_shape=crop_shape,
        distort_image=False)

    cifar_train = cifar_train.preprocess(train_preprocess_fn)

    _, cifar_test = cifar10_dataset.get_centralized_datasets(
        crop_shape=crop_shape, test_batch_size=10000)

    input_spec = cifar_train.create_tf_dataset_for_client(
        cifar_train.client_ids[0]).element_spec

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]
    model_builder = lambda: simple_cnn(crop_shape, 10, add_softmax=True)

    def tff_model_fn():
        return tff.learning.from_keras_model(
            keras_model=model_builder(),
            input_spec=input_spec,
            loss=loss_builder(),
            metrics=metrics_builder(),
        )

    iterative_process = iterative_process_builder(tff_model_fn)

    train_client_datasets_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            sample_range=cifar_train.client_ids, random_seed=FLAGS.seed),
        size=FLAGS.cohort_size,
    )

    evaluate_fn = tff.learning.build_federated_evaluation(
        tff_model_fn, use_experimental_simulation_loop=True)

    def evaluation_fn(model_weights, round_num):
        del round_num
        return evaluate_fn(model_weights, [cifar_test])

    logging.info(model_builder().summary())

    state = iterative_process.initialize()
    round_num = 0
    current_model = iterative_process.get_model_weights(state)

    loop_start_time = time.time()
    start_time_simulation = time.perf_counter()
    while round_num < FLAGS.central_num_iterations:
        data_prep_start_time = time.time()
        # federated_train_data = train_client_datasets_fn(round_num)
        federated_train_data = [
            cifar_train.create_tf_dataset_for_client(i)
            for i in train_client_datasets_fn(round_num)
        ]
        metrics = {"prepare_datasets_secs": time.time() - data_prep_start_time}

        res = iterative_process.next(state, federated_train_data)
        state = res.state
        round_metrics = res.metrics
        current_model = iterative_process.get_model_weights(state)
        metrics.update(round_metrics["client_work"]["train"])

        logging.info("Round {:2d}, {:.2f}s per round in average.".format(
            round_num, (time.time() - loop_start_time) / (round_num + 1)))

        if round_num % FLAGS.evaluation_frequency == 0:
            # Compute evaluation metrics
            eval_metrics = evaluation_fn(current_model, round_num)
            metrics.update({
                "central eval " + k: v
                for k, v in eval_metrics["eval"].items()
            })

        metrics["duration_of_iteration"] = time.time() - data_prep_start_time
        print(metrics)
        round_num += 1

    end_time = time.perf_counter()
    print("Total simulation time (s):", end_time - start_time_simulation)
    print("Total time (s):", end_time - start_time_total)


if __name__ == "__main__":
    app.run(main)
