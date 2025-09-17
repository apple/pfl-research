# Copyright © 2023-2024 Apple Inc.
import os
import pickle
import subprocess
import sys

import pytest

from pfl.internal.ops.common_ops import check_mlx_installed, get_pytorch_major_version, get_tf_major_version


class TestMultiProcess:
    """
    Test cases for checking that running with multiple processes
    will give the same results as 1 process.
    """

    @pytest.mark.parametrize('backend_tup', [
        pytest.param(('pytorch', 'torchrun --nproc_per_node=2'),
                     id='pytorch',
                     marks=[
                         pytest.mark.skipif(not get_pytorch_major_version(),
                                            reason='PyTorch not installed')
                     ]),
        pytest.param(('tensorflow', 'PFL_WORKER_RANK=0 python {} & PFL_WORKER_RANK=1 python {} && wait'),
                     id='tensorflow',
                     marks=[
                         pytest.mark.skipif(get_tf_major_version() < 2,
                                            reason='not tf>=2')
                     ]),
        pytest.param(
            ('mlx', 'mpirun -np 2 -x DYLD_LIBRARY_PATH=/opt/homebrew/lib'),
            id='mlx',
            marks=[
                pytest.mark.skipif(not check_mlx_installed(),
                                   reason='MLX not installed')
            ])
    ])
    def test_two_processes(self, tmp_path, ports, backend_tup,
                           check_equal_stats, check_equal_metrics):
        backend_framework, cmd_prefix = backend_tup

        single_worker_result_path = os.path.join(tmp_path, 'single.p')
        worker1_result_path = os.path.join(tmp_path, 'worker1.p')
        train_script_path = os.path.join(os.path.dirname(__file__),
                                         'run_training_on_fake_data.py')

        common_arguments = ['--local_num_epochs', '2', '--cohort_size', '3']

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
        
        if backend_framework == 'tensorflow':
            # For TensorFlow, we need to run two separate processes with different worker ranks
            cmd1 = [sys.executable, train_script_path,
                   '--output_path', worker1_result_path, '--backend_framework',
                   backend_framework, '--use_metric_spec',
                   str(True), *common_arguments]
            cmd2 = cmd1.copy()
            
            worker1_env['PFL_WORKER_RANK'] = '0'
            worker1_env['PFL_WORKER_ADDRESSES'] = 'localhost:12345,localhost:12346'
            worker2_env = os.environ.copy()
            worker2_env['PFL_WORKER_RANK'] = '1'
            worker2_env['PFL_WORKER_ADDRESSES'] = 'localhost:12345,localhost:12346'
            
            p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=worker1_env)
            p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=worker2_env)
            
            std1, err1 = p1.communicate()
            std2, err2 = p2.communicate()
            
        elif backend_framework == 'pytorch':
            # For PyTorch, use torchrun
            cmd_arguments = cmd_prefix.split() + [train_script_path,
                           '--output_path', worker1_result_path, '--backend_framework',
                           backend_framework, '--use_metric_spec',
                           str(True), *common_arguments]
            
            # Set up distributed environment for PyTorch
            worker1_env['PFL_WORKER_ADDRESSES'] = 'localhost:12345,localhost:12346'
            
            p1 = subprocess.Popen(cmd_arguments,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=worker1_env)
            std1, err1 = p1.communicate()
            
        else:
            # For other frameworks (like MLX), use the original approach
            cmd_arguments = [
                *cmd_prefix.split(), sys.executable, train_script_path,
                '--output_path', worker1_result_path, '--backend_framework',
                backend_framework, '--use_metric_spec',
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
