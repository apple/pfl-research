# Copyright © 2024 Apple Inc.
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
import time

import fedml
from compare_utils.pytorch import simple_cnn
from fedml import FedMLRunner
from fedml.data.cifar10.data_loader import load_partition_data_cifar10

if __name__ == "__main__":
    start_time_total = time.perf_counter()
    # init FedML framework
    args = fedml.init(should_init_logs=False)

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_partition_data_cifar10(
        args.dataset,
        args.data_cache_dir,
        args.partition_method,
        args.partition_alpha,
        args.client_num_in_total,
        args.batch_size,
    )

    # load model
    model = simple_cnn((32, 32, 3), 10, transpose=False)

    start_time_simulation = time.perf_counter()

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()

    end_time = time.perf_counter()
    print("Total simulation time (s):", end_time - start_time_simulation)
    print("Total time (s):", end_time - start_time_total)
