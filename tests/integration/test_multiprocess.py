# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import os
import pickle
import subprocess
import sys

import pytest

from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version


@pytest.mark.is_slow
@pytest.mark.horovod
class TestMultiProcess:
    """
    Test cases for checking that running with multiple processes with Horovod
    will give the same results as 1 process.
    """

    @pytest.mark.parametrize('backend_framework', [
        pytest.param('pytorch',
                     id='pytorch',
                     marks=[
                         pytest.mark.skipif(not get_pytorch_major_version(),
                                            reason='PyTorch not installed')
                     ]),
        pytest.param('tensorflow',
                     id='tensorflow',
                     marks=[
                         pytest.mark.skipif(get_tf_major_version() < 2,
                                            reason='not tf>=2')
                     ]),
    ])
    def test_two_processes(self, tmp_path, ports, backend_framework,
                           check_equal_stats, check_equal_metrics):

        single_worker_result_path = os.path.join(tmp_path, 'single.p')
        worker1_result_path = os.path.join(tmp_path, 'worker1.p')
        train_script_path = os.path.join(os.path.dirname(__file__),
                                         'run_training_on_fake_data.py')

        common_arguments = ['--local_num_epochs', '2', '--cohort_size', '10']

        # Run `run_training_on_fake_data.py` with a single worker.
        cmd_arguments = [
            sys.executable, train_script_path, '--output_path',
            single_worker_result_path, '--backend_framework',
            backend_framework, '--use_metric_spec',
            str(True), *common_arguments
        ]
        p = subprocess.Popen(cmd_arguments,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=os.environ.copy())
        _, err = p.communicate()
        # Load results from the pickle dumped by the other process.
        try:
            with open(single_worker_result_path, 'rb') as f:
                stats_single, metrics_single = pickle.load(f)
        except:
            raise AssertionError(
                f'Single worker process failed, did not dump statistics to disk. Error: {err}'
            )

        # Run `run_training_on_fake_data.py` with two workers.
        worker1_env = os.environ.copy()
        cmd_arguments = [
            'horovodrun', '--gloo', '-np', '2', '-H', 'localhost:2',
            sys.executable, train_script_path, '--output_path',
            worker1_result_path, '--backend_framework', backend_framework,
            '--use_metric_spec',
            str(True), *common_arguments
        ]
        p1 = subprocess.Popen(cmd_arguments,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=worker1_env)

        std1, err1 = p1.communicate()
        # Load results from the pickles dumped by the other processes.
        try:
            with open(f'{worker1_result_path}.0', 'rb') as f:
                stats_worker1, metrics_worker1 = pickle.load(f)
        except:
            raise AssertionError(
                f'Worker process 1 failed, did not dump statistics to disk. Stdout: {std1} \n Error: {err1}'
            )

        try:
            with open(f'{worker1_result_path}.1', 'rb') as f:
                stats_worker2, metrics_worker2 = pickle.load(f)
        except:
            raise AssertionError(
                f'Worker process 2 failed, did not dump statistics to disk. Stdout: {std1} \n Error: {err1}'
            )

        for metrics_description, metrics in [
            ('single', metrics_single),
            ('worker1', metrics_worker1),
            ('worker2', metrics_worker2),
        ]:
            print(f'Metrics from {metrics_description}:')
            for (metric_name, metric_value) in metrics:
                print(f'  {metric_name}: {metric_value}')

        # The workers should have exchanged metrics and summed them.
        # Both of them should therefore have the overal metrics and
        # statistics.
        check_equal_metrics(metrics_worker1, metrics_worker2)
        check_equal_metrics(metrics_single, metrics_worker1)

        check_equal_stats(stats_worker1, stats_worker2)
        check_equal_stats(stats_single, stats_worker1)
