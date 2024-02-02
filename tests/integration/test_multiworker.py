# Copyright © 2023-2024 Apple Inc.
import os
import pickle
import subprocess
import sys

import pytest

from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version


def _run_test_two_workers(tmp_path,
                          ports,
                          backend_framework,
                          local_privacy_mechanism,
                          central_privacy_mechanism,
                          check_equal_metrics,
                          check_equal_stats,
                          use_metric_spec=False,
                          use_framework_dataset=False):
    single_worker_result_path = os.path.join(tmp_path, 'single.p')
    worker1_result_path = os.path.join(tmp_path, 'worker1.p')
    train_script_path = os.path.join(os.path.dirname(__file__),
                                     'run_training_on_fake_data.py')

    def make_cmd_arguments(output_path):
        return [
            sys.executable, train_script_path, '--local_num_epochs', '2',
            '--cohort_size', '3', '--backend_framework', backend_framework,
            '--use_metric_spec',
            str(use_metric_spec), '--local_privacy_mechanism',
            local_privacy_mechanism, '--central_privacy_mechanism',
            central_privacy_mechanism, '--use_framework_dataset',
            str(use_framework_dataset), '--output_path', output_path
        ]

    env = os.environ.copy()
    # Run `run_training_on_fake_data.py` with a single worker.
    p = subprocess.Popen(make_cmd_arguments(single_worker_result_path),
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         env=env)
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
    worker1_env['PFL_WORKER_RANK'] = '0'
    worker1_env['PFL_WORKER_ADDRESSES'] = ",".join(
        [f'localhost:{p}' for p in ports])
    p1 = subprocess.Popen(make_cmd_arguments(worker1_result_path),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=worker1_env)

    worker2_env = os.environ.copy()
    worker2_env['PFL_WORKER_RANK'] = '1'
    worker2_env['PFL_WORKER_ADDRESSES'] = ",".join(
        [f'localhost:{p}' for p in ports])
    p2 = subprocess.Popen(make_cmd_arguments(worker1_result_path),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=worker2_env)

    std1, err1 = p1.communicate()
    std2, err2 = p2.communicate()
    # Load results from the pickles dumped by the other processes.
    try:
        with open(f'{worker1_result_path}.0', 'rb') as f:
            stats_worker1, metrics_worker1 = pickle.load(f)
    except:
        raise AssertionError(
            'Worker process 1 failed, did not dump statistics to disk. worker1_stdout: {} \n worker1_error: {} worker2_stdout: {} \n worker2_error: {} '
            .format(std1, err1, std2, err2))

    try:
        with open(f'{worker1_result_path}.1', 'rb') as f:
            stats_worker2, metrics_worker2 = pickle.load(f)
    except:
        raise AssertionError(
            'Worker process 2 failed, did not dump statistics to disk. worker1_stdout: {} \n worker1_error: {} worker2_stdout: {} \n worker2_error: {} '
            .format(std1, err1, std2, err2))

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


class TestMultiWorker:
    """
    Test cases for checking that running with multiple workers
    will give the same results as 1 worker.
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
    @pytest.mark.parametrize('use_metric_spec', [False, True])
    @pytest.mark.parametrize(
        'local_privacy_mechanism,central_privacy_mechanism',
        [('none', 'none'), ('gaussian', 'none'), ('none', 'gaussian')])
    def test_two_workers(self, tmp_path, ports, backend_framework,
                         use_metric_spec, local_privacy_mechanism,
                         central_privacy_mechanism, check_equal_metrics,
                         check_equal_stats):

        _run_test_two_workers(tmp_path,
                              ports,
                              backend_framework,
                              local_privacy_mechanism,
                              central_privacy_mechanism,
                              check_equal_metrics,
                              check_equal_stats,
                              use_metric_spec=use_metric_spec)

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
    def test_two_workers_framework_dataset(self, tmp_path, ports,
                                           backend_framework,
                                           check_equal_metrics,
                                           check_equal_stats):
        _run_test_two_workers(tmp_path,
                              ports,
                              backend_framework,
                              local_privacy_mechanism='gaussian',
                              central_privacy_mechanism='gaussian',
                              check_equal_metrics=check_equal_metrics,
                              check_equal_stats=check_equal_stats,
                              use_metric_spec=True,
                              use_framework_dataset=True)
