# Copyright © 2023-2024 Apple Inc.
import cProfile
import json
import operator
import os
from glob import glob
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest

from pfl.algorithm.federated_averaging import FederatedAveraging
from pfl.callback import (
    AggregateMetricsToDisk,
    CentralEvaluationCallback,
    CentralEvaluationWithEMACallback,
    CheckpointPolicy,
    ConvergenceCallback,
    EarlyStoppingCallback,
    MetricImprovementCheckpointPolicy,
    ModelCheckpointingCallback,
    PolicyBasedModelCheckpointingCallback,
    ProfilerCallback,
    RestoreTrainingCallback,
    StopwatchCallback,
    TensorBoardCallback,
    TrackBestOverallMetrics,
    WandbCallback,
)
from pfl.common_types import Population, Saveable
from pfl.data.dataset import Dataset
from pfl.hyperparam import ModelHyperParams
from pfl.internal.ops import get_tf_major_version
from pfl.internal.platform.generic_platform import GenericPlatform
from pfl.metrics import MetricName, Metrics, StringMetricName, Weighted, get_overall_value
from pfl.model.base import StatefulModel
from pfl.model.ema import CentralExponentialMovingAverage

# pylint: disable=too-many-lines


@pytest.fixture(scope='module')
def central_dataset():
    return Dataset((np.random.normal(size=(10, 2)), ))


# pytype: disable=wrong-arg-count
loss_key = MetricName('loss', Population.VAL)
datapoint_key = MetricName('number of data points', Population.TRAIN)
# pytype: enable=wrong-arg-count


@pytest.fixture(scope='function')
def metrics():
    return Metrics([(loss_key, Weighted(20, 2)), (datapoint_key, 2)])


@pytest.fixture
def algorithm():
    return FederatedAveraging()


@pytest.fixture
def mock_saveable():
    return MagicMock(spec=Saveable)


# pylint: disable=protected-access
class TestRestoreTrainingCallback:

    @pytest.fixture
    def callback(self, algorithm, mock_saveable, tmp_path):
        return RestoreTrainingCallback([algorithm, mock_saveable],
                                       tmp_path,
                                       checkpoint_frequency=2)

    def test_on_train_begin_no_state(self, callback, algorithm, mock_model,
                                     nn_train_params):
        callback.on_train_begin(model=mock_model)
        mock_model.load.assert_not_called()
        assert algorithm._current_central_iteration == 0

    def test_save_restore(self, callback, algorithm, mock_saveable, mock_model,
                          nn_train_params, tmp_path):
        iteration = 2
        algorithm._current_central_iteration = iteration
        stop, metrics_out = callback.after_central_iteration(
            Metrics(), mock_model, central_iteration=iteration)
        assert not stop
        assert not len(metrics_out)

        mock_saveable.save.assert_called_once_with(str(tmp_path))

        with open(os.path.join(tmp_path, 'algorithm_checkpoint.json')) as f:
            state = json.load(f)
            assert state == {'current_central_iteration': iteration}

    def test_after_central_iteration_skip(self, callback, mock_model,
                                          mock_saveable, nn_train_params):
        callback.after_central_iteration(Metrics(),
                                         mock_model,
                                         central_iteration=1)
        mock_saveable.save.assert_not_called()

    def test_restore_checkpoint(self, callback, algorithm, mock_saveable,
                                mock_model, nn_train_params, tmp_path):
        assert algorithm._current_central_iteration == 0
        algorithm._current_central_iteration = 2
        callback.after_central_iteration(Metrics(),
                                         mock_model,
                                         central_iteration=2)
        callback.on_train_begin(model=mock_model)
        assert algorithm._current_central_iteration == 2
        mock_saveable.load.assert_called_once_with(str(tmp_path))


# pylint: enable=protected-access


class TestCentralEvaluationCallback:

    @pytest.mark.parametrize(
        'distribute_evaluation,allow_disteval,local_batch_size', [
            (False, False, None),
            (False, True, None),
            pytest.param(True,
                         False,
                         None,
                         marks=pytest.mark.xfail(raises=AssertionError,
                                                 reason='not supported')),
            (True, True, None),
            (True, True, 2),
        ])
    @pytest.mark.parametrize(
        'format_fn', [None, lambda s: StringMetricName(f'central metric {s}')])
    def test_after_central_iteration(self, distribute_evaluation,
                                     allow_disteval, local_batch_size,
                                     format_fn, central_dataset, metrics,
                                     mock_model):
        mock_model_params = MagicMock(spec=ModelHyperParams,
                                      local_batch_size=local_batch_size)
        mock_model.allows_distributed_evaluation = allow_disteval

        cb = CentralEvaluationCallback(
            central_dataset,
            mock_model_params,
            2,
            distribute_evaluation=distribute_evaluation,
            format_fn=format_fn)
        stop_signal, returned_metrics = (cb.after_central_iteration(
            metrics, mock_model, central_iteration=2))

        assert not stop_signal
        # Two metrics added from evaluation.
        assert len(returned_metrics) == 2
        if format_fn is None:
            assert get_overall_value(
                returned_metrics[StringMetricName('Central val | loss')]) == 1
            assert get_overall_value(returned_metrics[StringMetricName(
                'Central val | number of data points')]) == 10
        else:
            assert get_overall_value(
                returned_metrics[StringMetricName('central metric loss')]) == 1
            assert get_overall_value(returned_metrics[StringMetricName(
                'central metric number of data points')]) == 10

        assert mock_model_params.local_batch_size == local_batch_size

    def test_after_central_iteration_skip(self, central_dataset, metrics,
                                          mock_model):
        mock_model_params = MagicMock(spec=ModelHyperParams,
                                      local_batch_size=1)
        cb = CentralEvaluationCallback(central_dataset, mock_model_params, 2)
        stop_signal, returned_metrics = (cb.after_central_iteration(
            metrics, mock_model, central_iteration=1))

        assert not stop_signal
        # No metrics because evaluation should have been skipped.
        assert len(returned_metrics) == 0


class TestCentralEvaluationWithEMACallback:

    @pytest.mark.parametrize('distribute_evaluation,allow_disteval', [
        (False, False),
        (False, True),
        pytest.param(True,
                     False,
                     marks=pytest.mark.xfail(raises=AssertionError,
                                             reason='not supported')),
        (True, True),
    ])
    @pytest.mark.parametrize(
        'format_fn',
        [None, lambda s: StringMetricName(f'central metric EMA {s}')])
    def test_after_central_iteration(self, distribute_evaluation,
                                     allow_disteval, format_fn,
                                     central_dataset, metrics, mock_model):
        mock_model.allows_distributed_evaluation = allow_disteval
        ema = CentralExponentialMovingAverage(mock_model, 0.9)
        mock_model_params = MagicMock(spec=ModelHyperParams,
                                      local_batch_size=1)
        cb = CentralEvaluationWithEMACallback(
            central_dataset,
            ema,
            mock_model_params,
            2,
            distribute_evaluation=distribute_evaluation,
            format_fn=format_fn)
        stop_signal, returned_metrics = (cb.after_central_iteration(
            metrics, mock_model, central_iteration=2))

        assert not stop_signal
        # Two metrics added from evaluation.
        assert len(returned_metrics) == 2
        if format_fn is None:
            assert get_overall_value(returned_metrics[StringMetricName(
                'Central val EMA | loss')]) == 1
            assert get_overall_value(returned_metrics[StringMetricName(
                'Central val EMA | number of data points')]) == 10
        else:
            assert get_overall_value(returned_metrics[StringMetricName(
                'central metric EMA loss')]) == 1
            assert get_overall_value(returned_metrics[StringMetricName(
                'central metric EMA number of data points')]) == 10

    def test_after_central_iteration_skip(self, central_dataset, metrics,
                                          mock_model):
        ema = CentralExponentialMovingAverage(mock_model, 0.9)
        mock_model_params = MagicMock(spec=ModelHyperParams,
                                      local_batch_size=1)
        cb = CentralEvaluationWithEMACallback(central_dataset, ema,
                                              mock_model_params, 2)
        stop_signal, returned_metrics = (cb.after_central_iteration(
            metrics, mock_model, central_iteration=1))

        assert not stop_signal
        # No metrics because evaluation should have been skipped.
        assert len(returned_metrics) == 0


class TestConvergenceCallback:

    def test_converge(self, metrics, mock_model):

        early_stopping_cb = ConvergenceCallback(
            metric_name='val population | loss',
            patience=3,
            performance_threshold=11,
            performance_is_better=operator.lt)
        # Calling 2 times should not trigger stop
        for _ in range(2):
            stop_signal, returned_metrics = (
                early_stopping_cb.after_central_iteration(metrics,
                                                          mock_model,
                                                          central_iteration=1))
            assert not stop_signal
            assert len(returned_metrics) == 0

        # Calling 3rd time should trigger stop
        stop_signal, returned_metrics = (
            early_stopping_cb.after_central_iteration(metrics,
                                                      mock_model,
                                                      central_iteration=1))
        assert stop_signal
        assert len(returned_metrics) == 1
        # Convergence started after 2 points (in 1st iteration)
        assert get_overall_value(returned_metrics[StringMetricName(
            'data points for convergence')]) == 2

    def test_no_converge(self, metrics, mock_model):

        early_stopping_cb = ConvergenceCallback(
            metric_name='val population | loss',
            patience=3,
            performance_threshold=9,
            performance_is_better=operator.lt)

        for _ in range(10):
            # Should never trigger stop with loss 10 and threshold 9
            # when lower is better.
            stop_signal, returned_metrics = (
                early_stopping_cb.after_central_iteration(metrics,
                                                          mock_model,
                                                          central_iteration=1))
            assert not stop_signal
            assert len(returned_metrics) == 0


class TestEarlyStoppingCallback:

    def test_stable_metric(self, metrics, mock_model):

        early_stopping_cb = EarlyStoppingCallback(metric_name=loss_key,
                                                  patience=3)
        # Calling 3 times should not trigger stop (2 rounds of not improving).
        for _ in range(3):
            stop_signal, returned_metrics = (
                early_stopping_cb.after_central_iteration(metrics,
                                                          mock_model,
                                                          central_iteration=1))
            assert not stop_signal
            assert len(returned_metrics) == 0

        # The 3rd round of not improving should trigger stop signal.
        stop_signal, returned_metrics = (
            early_stopping_cb.after_central_iteration(metrics,
                                                      mock_model,
                                                      central_iteration=1))
        assert stop_signal
        assert len(returned_metrics) == 0

    def test_stop_6th_iteration(self, metrics, mock_model):
        loss_sequence = [4, 3, 2, 1, 2, 3, 2]

        for metric_name in [loss_key, 'val population | loss']:
            early_stopping_cb = EarlyStoppingCallback(metric_name, patience=3)
            for i, loss in enumerate(loss_sequence):
                metrics = Metrics([(loss_key, Weighted.from_unweighted(loss))])
                stop_signal, returned_metrics = (
                    early_stopping_cb.after_central_iteration(
                        metrics, mock_model, central_iteration=1))

                if i == 6:
                    # should stop at 7th iteration.
                    assert stop_signal
                else:
                    assert not stop_signal
                assert len(returned_metrics) == 0

    def test_improving_metric(self, mock_model):

        early_stopping_cb = EarlyStoppingCallback(
            metric_name=loss_key,
            patience=3,
            performance_is_better=operator.gt)

        for i in range(10):
            # Increasing metric is good, should therefore never trigger stop.
            metrics = Metrics([(loss_key, Weighted.from_unweighted(i))])
            stop_signal, returned_metrics = (
                early_stopping_cb.after_central_iteration(metrics,
                                                          mock_model,
                                                          central_iteration=1))
            assert not stop_signal
            assert len(returned_metrics) == 0


class TestTimerCallback:

    def test_after_central_iteration(self, metrics, mock_model):
        callback = StopwatchCallback()

        with patch('time.time', return_value=1.0):
            callback.on_train_begin(model=mock_model)

        with patch('time.time', return_value=2.0):
            stop_signal, returned_metrics = (callback.after_central_iteration(
                metrics, mock_model, central_iteration=2))

        assert not stop_signal
        # Three metrics added from evaluation.
        assert len(returned_metrics) == 3
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall time elapsed (min)')]) == 0.02
        assert get_overall_value(returned_metrics[StringMetricName(
            'duration of iteration (s)')]) == 1
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall average duration of iteration (s)')]) == 1

        with patch('time.time', return_value=4.0):
            stop_signal, returned_metrics = (callback.after_central_iteration(
                metrics, mock_model, central_iteration=2))

        assert not stop_signal
        # Three metrics added from evaluation.
        assert len(returned_metrics) == 3
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall time elapsed (min)')]) == 0.05
        assert get_overall_value(returned_metrics[StringMetricName(
            'duration of iteration (s)')]) == 2
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall average duration of iteration (s)')]) == 1.5

    def test_after_central_iteration_minutes(self, metrics, mock_model):
        callback = StopwatchCallback(measure_round_in_minutes=True)

        with patch('time.time', return_value=1.0):
            callback.on_train_begin(model=mock_model)

        with patch('time.time', return_value=2.0):
            callback.after_central_iteration(metrics,
                                             mock_model,
                                             central_iteration=2)
        with patch('time.time', return_value=4.0):
            stop_signal, returned_metrics = (callback.after_central_iteration(
                metrics, mock_model, central_iteration=2))

        assert not stop_signal
        # Three metrics added from evaluation.
        assert len(returned_metrics) == 3
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall time elapsed (min)')]) == 0.05
        assert get_overall_value(returned_metrics[StringMetricName(
            'duration of iteration (min)')]) == 0.03
        assert get_overall_value(returned_metrics[StringMetricName(
            'overall average duration of iteration (min)')]) == 0.03


@pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
class TestTensorBoardCallback:

    @pytest.fixture(scope='module')
    def mock_model_params(self):
        return MagicMock(spec=ModelHyperParams, current_central_iteration=0)

    @pytest.fixture(scope='module')
    def tf(self):
        import tensorflow as tf
        return tf

    @pytest.fixture
    def summary_iterator(self):
        from tensorflow.python.summary.summary_iterator import summary_iterator
        return summary_iterator

    @pytest.fixture(scope='module')
    def model(self, tf, mock_model_params, user_dataset):
        from pfl.model.tensorflow import TFModel
        keras_model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(2, input_shape=(2, ))])
        keras_model.compile(loss='mse')

        # Dummy eval just to cache the forward graph.
        model = TFModel(model=keras_model, metrics={}, central_optimizer=None)
        model.evaluate(user_dataset, mock_model_params)

        return model

    def test_write_graph(self, tensorflow_ops, tmp_path, tf, model):
        from tensorflow.python.summary.summary_iterator import summary_iterator
        callback = TensorBoardCallback(tmp_path,
                                       write_weights=False,
                                       write_graph=True)
        callback.after_central_iteration(Metrics(), model, central_iteration=1)
        events = list(summary_iterator(glob(str(tmp_path / 'train/*'))[0]))
        assert len(events) == 2
        assert isinstance(events[1].graph_def, bytes)

    def test_after_central_iteration_scalars(self, tmp_path, tf,
                                             summary_iterator,
                                             mock_model_params, model):

        callback = TensorBoardCallback(tmp_path,
                                       write_weights=False,
                                       write_graph=False)
        metrics = Metrics()

        # This metric should not be included.
        ignore_serialize_metric_name = MagicMock(ignore_serialization=True,
                                                 spec=StringMetricName)
        ignore_serialize_metric_name.__str__.return_value = 'ignored_metric'
        metrics[ignore_serialize_metric_name] = 1
        metrics[StringMetricName("loss")] = 0.125
        metrics[MetricName("accuracy", Population.TRAIN)] = 0.25
        metrics[MetricName("accuracy", Population.VAL)] = 0.5
        stop, metrics_out = callback.after_central_iteration(
            metrics, model, central_iteration=2)
        assert not stop
        assert not len(metrics_out)

        events = list(summary_iterator(glob(str(tmp_path / 'train/*'))[0]))
        assert len(events) == 3
        assert events[1].summary.value[0].tag == 'Loss'
        assert tf.make_ndarray(events[1].summary.value[0].tensor) == 0.125
        assert events[2].summary.value[0].tag == 'accuracy'
        assert tf.make_ndarray(events[2].summary.value[0].tensor) == 0.25

        events = list(summary_iterator(glob(str(tmp_path / 'val/*'))[0]))
        assert len(events) == 2
        assert events[1].summary.value[0].tag == 'accuracy'
        assert tf.make_ndarray(events[1].summary.value[0].tensor) == 0.5

    def test_after_central_iteration_histograms(self, tmp_path, tf,
                                                summary_iterator,
                                                mock_model_params, model):

        callback = TensorBoardCallback(tmp_path,
                                       write_weights=True,
                                       write_graph=False)
        metrics = Metrics()
        stop, metrics_out = callback.after_central_iteration(
            metrics, model, central_iteration=2)
        assert not stop
        assert not len(metrics_out)

        events = list(summary_iterator(glob(str(tmp_path / 'train/*'))[0]))
        assert len(events) == 3
        assert events[1].summary.value[0].tag == 'dense/kernel_0/histogram'
        assert events[2].summary.value[0].tag == 'dense/bias_0/histogram'

    @patch('subprocess.Popen')
    def test_host_tensorboard(self, mock_popen, tmp_path, tf, summary_iterator,
                              mock_model_params, model):

        callback = TensorBoardCallback(tmp_path,
                                       write_weights=False,
                                       write_graph=False)
        callback.on_train_begin(model=model)
        assert mock_popen.call_count == 0

        port = 1337
        callback = TensorBoardCallback(tmp_path,
                                       write_weights=False,
                                       write_graph=False,
                                       tensorboard_port=port)
        callback.on_train_begin(model=model)
        mock_popen.assert_called_once_with([
            'tensorboard', '--logdir', tmp_path, '--port',
            str(port), '--bind_all'
        ],
                                           stdout=-1,
                                           stderr=-1,
                                           env=ANY)

    def test_on_train_end(self, tmp_path, model, mock_model_params):
        with patch(
                'tensorflow.summary.create_file_writer') as create_mock_writer:
            mock_writer = MagicMock()
            create_mock_writer.return_value = mock_writer
            callback = TensorBoardCallback(tmp_path,
                                           write_weights=False,
                                           write_graph=False)
            callback.on_train_end(model=model)
            assert mock_writer.close.call_count == 3


@pytest.mark.parametrize('checkpoint_frequency,expected_call_count,numbered', [
    (0, 1, True),
    (1, 2, True),
    (2, 1, True),
    (0, 1, False),
    (1, 2, False),
    (2, 1, False),
])
def test_model_checkpointing_callback(checkpoint_frequency,
                                      expected_call_count, numbered, tmp_path):
    platform = MagicMock(spec=GenericPlatform)
    platform.create_checkpoint_directories.side_effect = lambda dirs: dirs
    model = MagicMock(spec=StatefulModel)
    with patch('pfl.internal.platform.selector.get_platform',
               return_value=platform):
        callback = ModelCheckpointingCallback(
            str(tmp_path),
            checkpoint_frequency=checkpoint_frequency,
            numbered=numbered)
        callback.after_central_iteration(Metrics(), model, central_iteration=0)
        callback.after_central_iteration(Metrics(), model, central_iteration=1)
        callback.on_train_end(model=model)
    assert model.save.call_count == expected_call_count

    call_args_list = model.save.call_args_list
    if numbered:
        if checkpoint_frequency != 0:
            for idx in range(model.save.call_count):
                central_iteration = (idx + 1) * checkpoint_frequency - 1
                assert call_args_list[idx].args == (
                    f'{tmp_path}/{central_iteration:05}', )
        else:
            assert call_args_list[0].args == (f'{tmp_path}/final', )
    else:
        for call_args in model.save.call_args_list:
            assert call_args.args == (f'{tmp_path}', )


@pytest.mark.parametrize('policy_results,should_checkpoint_at_end,numbered', [
    ([False, False], False, False),
    ([True, False], False, False),
    ([False, True], False, False),
    ([True, True], False, False),
    ([False, False], False, True),
    ([True, False], False, True),
    ([False, True], False, True),
    ([True, True], False, True),
    ([False, False], True, False),
    ([True, False], True, False),
    ([False, True], True, False),
    ([True, True], True, False),
    ([False, False], True, True),
    ([True, False], True, True),
    ([False, True], True, True),
    ([True, True], True, True),
])
def test_policy_based_model_checkpointing_callback(policy_results,
                                                   should_checkpoint_at_end,
                                                   numbered, tmp_path):
    platform = MagicMock(spec=GenericPlatform)
    platform.create_checkpoint_directories.side_effect = lambda dirs: dirs
    model = MagicMock(spec=StatefulModel)
    policy = MagicMock(spec=CheckpointPolicy)
    policy.should_checkpoint_now.side_effect = policy_results
    policy.should_checkpoint_at_end.return_value = should_checkpoint_at_end
    with patch('pfl.internal.platform.selector.get_platform',
               return_value=platform):
        callback = PolicyBasedModelCheckpointingCallback(
            str(tmp_path), checkpoint_policy=policy, numbered=numbered)
        callback.after_central_iteration(Metrics(), model, central_iteration=0)
        callback.after_central_iteration(Metrics(), model, central_iteration=1)
        callback.on_train_end(model=model)

    call_args_list = model.save.call_args_list

    expected_call_count = should_checkpoint_at_end + sum(policy_results)
    assert model.save.call_count == expected_call_count

    if numbered:
        call_args_iter = iter(call_args_list)
        for central_iteration, checkpointed in enumerate(policy_results):
            if checkpointed:
                assert next(call_args_iter).args == (
                    f'{tmp_path}/{central_iteration:05}', )
        if should_checkpoint_at_end:
            assert next(call_args_iter).args == (f'{tmp_path}/final', )
    else:
        for call_args in model.save.call_args_list:
            assert call_args.args == (f'{tmp_path}', )


@pytest.mark.parametrize(
    'metric_values,threshold,expected_call_count,numbered', [
        ([3, 2, 1], 2, 1, False),
        ([1, 0, 0], 2, 2, False),
        ([0, 0, 0], 2, 1, False),
        ([3, 2, 1], 2, 1, True),
        ([1, 0, 0], 2, 2, True),
        ([0, 0, 0], 1, 1, True),
        ([3, 2, 1], None, 3, False),
        ([1, 0, 0], None, 2, False),
        ([0, 0, 0], None, 1, False),
        ([3, 2, 1], None, 3, True),
        ([1, 0, 0], None, 2, True),
        ([0, 0, 0], None, 1, True),
    ])
def test_metric_improvement_model_checkpointing_callback(
        metric_values, threshold, expected_call_count, numbered, tmp_path):
    platform = MagicMock(spec=GenericPlatform)
    platform.create_checkpoint_directories.side_effect = lambda dirs: dirs
    model = MagicMock(spec=StatefulModel)
    policy = MetricImprovementCheckpointPolicy(metric_name='metric_name',
                                               threshold_value=threshold)
    with patch('pfl.internal.platform.selector.get_platform',
               return_value=platform):
        callback = PolicyBasedModelCheckpointingCallback(
            str(tmp_path), checkpoint_policy=policy, numbered=False)
        for central_iteration in range(3):
            callback.after_central_iteration(
                Metrics({'metric_name':
                         metric_values[central_iteration]}.items()),
                model,
                central_iteration=central_iteration)
        callback.on_train_end(model=model)

    call_args_list = model.save.call_args_list

    assert model.save.call_count == expected_call_count

    if numbered:
        call_args_iter = iter(call_args_list)
        for central_iteration, (lhs, rhs) in enumerate(
                zip(metric_values[:-1], metric_values[1:])):
            if policy.performance_is_better(lhs, rhs):
                assert next(call_args_iter).args == (
                    f'{tmp_path}/{central_iteration:05}', )
    else:
        for call_args in model.save.call_args_list:
            assert call_args.args == (f'{tmp_path}', )


class TestProfilerCallback:

    @patch.object(cProfile.Profile, 'enable')
    @patch.object(cProfile.Profile, 'disable')
    @patch('pfl.callback.ProfilerCallback.platform')
    @pytest.mark.parametrize(
        'frequency, warmup_iterations, num_iterations, expected',
        [(1, 0, 10, list(range(10))), (2, 3, 10, [3, 5, 7, 9]),
         (2, 0, 10, [0, 2, 4, 6, 8]), (2, 11, 10, []), (5, 5, 3, [])])
    def test_per_iteration_callback(self, mock_get_platform, disable_mock,
                                    enable_mock, nn_train_params, mock_model,
                                    tmp_path, frequency, warmup_iterations,
                                    num_iterations, expected):
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform
        mock_platform.create_checkpoint_directories.return_value = [tmp_path]

        cb = ProfilerCallback(frequency=frequency,
                              warmup_iterations=warmup_iterations)
        cb.on_train_begin(model=mock_model)
        if warmup_iterations == 0:
            enable_mock.assert_called_once()
        else:
            enable_mock.assert_not_called()

        for central_iteration in range(num_iterations):
            cb.after_central_iteration(Metrics(),
                                       mock_model,
                                       central_iteration=central_iteration)
            if (central_iteration + 1) in expected:
                assert enable_mock.call_count == expected.index(
                    central_iteration + 1) + 1
            if central_iteration in expected:
                assert disable_mock.call_count == expected.index(
                    central_iteration) + 1

        cb.on_train_end(model=mock_model)

        profiles = os.listdir(tmp_path)
        assert len(profiles) == len(expected)
        expected_profiles = {f'profile-iteration-{i}.cprof' for i in expected}
        assert expected_profiles == set(profiles)

    @patch.object(cProfile.Profile, 'enable')
    @patch.object(cProfile.Profile, 'disable')
    @patch('pfl.callback.ProfilerCallback.platform')
    @pytest.mark.parametrize(
        'frequency, warmup_iterations, num_iterations, expected',
        [(None, 0, 10, 1), (None, 3, 5, 1), (None, 10, 5, 0)])
    def test_all_iterations_callback(self, mock_get_platform, disable_mock,
                                     enable_mock, nn_train_params, mock_model,
                                     tmp_path, frequency, warmup_iterations,
                                     num_iterations, expected):
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform
        mock_platform.create_checkpoint_directories.return_value = [tmp_path]

        cb = ProfilerCallback(frequency=frequency,
                              warmup_iterations=warmup_iterations)
        cb.on_train_begin(model=mock_model)
        if warmup_iterations == 0:
            enable_mock.assert_called_once()
        else:
            enable_mock.assert_not_called()

        for central_iteration in range(num_iterations):
            cb.after_central_iteration(Metrics(),
                                       mock_model,
                                       central_iteration=central_iteration)
            if central_iteration == warmup_iterations - 1:
                enable_mock.assert_called_once()

        disable_mock.assert_not_called()
        cb.on_train_end(model=mock_model)
        if warmup_iterations <= num_iterations:
            disable_mock.assert_called_once()
            enable_mock.assert_called_once()

        profiles = os.listdir(tmp_path)
        assert len(profiles) == expected
        if expected:
            assert profiles[0] == 'profile.cprof'


class TestAggregateMetricsToDisk:

    @patch('pfl.callback.AggregateMetricsToDisk.platform')
    @pytest.mark.parametrize('frequency,expected_lines', [(1, [
        'central_iteration,M1,M2,M3\n',
        '0,0,1,\n',
        '1,,3,\n',
        '2,4,,5\n',
    ]), (2, [
        'central_iteration,M1,M2,M3\n',
        '0,0,1,\n',
        '2,4,,5\n',
    ])])
    def test_callback(self, mock_get_platform, frequency, expected_lines,
                      mock_model, tmp_path):
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform
        mock_platform.create_checkpoint_directories.return_value = [tmp_path]

        out_path = os.path.join(tmp_path, 'out.csv')
        cb = AggregateMetricsToDisk(out_path, frequency=frequency)
        cb.on_train_begin(model=mock_model)

        cb.after_central_iteration(Metrics([
            ('m1', 0),
            ('m2', 1),
        ]),
                                   mock_model,
                                   central_iteration=0)
        cb.after_central_iteration(Metrics([
            ('m2', 3),
        ]),
                                   mock_model,
                                   central_iteration=1)
        cb.after_central_iteration(Metrics([
            ('m1', 4),
            ('m3', 5),
        ]),
                                   mock_model,
                                   central_iteration=2)
        cb.on_train_end(model=mock_model)

        with open(out_path) as f:
            for line, expected_line in zip(f.readlines(), expected_lines):
                assert line == expected_line


class TestTrackBestOverallMetrics:

    def test_callback(self, mock_model):
        loss_name = MetricName('loss', Population.TRAIN)
        accuracy_name = MetricName('accuracy', Population.TRAIN)
        cb = TrackBestOverallMetrics([
            loss_name,
        ], ['train population | accuracy'])

        no_metrics = cb.on_train_begin(model=mock_model)
        assert len(no_metrics) == 0

        observed_losses = [3, 2, 1, 2, 3, 0.5]
        observed_accs = [0.8, 0.7, 0.7, 0.81, 0.9, 0.0]
        expected_best_losses = [3, 2, 1, 1, 1, 0.5]
        expected_best_accs = [0.8, 0.8, 0.8, 0.81, 0.9, 0.9]

        should_stop, no_metrics = cb.after_central_iteration(
            Metrics(), mock_model, central_iteration=0)
        assert not should_stop
        assert len(no_metrics) == 0

        for i, (observed_loss, observed_acc, expected_best_loss,
                expected_best_acc) in enumerate(
                    zip(observed_losses, observed_accs, expected_best_losses,
                        expected_best_accs)):
            aggregate_metrics = Metrics([(loss_name, observed_loss),
                                         (accuracy_name, observed_acc)])
            should_stop, best_metrics = cb.after_central_iteration(
                aggregate_metrics, mock_model, central_iteration=i)
            assert not should_stop
            assert best_metrics.to_simple_dict() == {
                'Train population | loss | best overall': expected_best_loss,
                'Train population | accuracy | best overall':
                expected_best_acc,
            }

        cb.after_central_iteration(Metrics(), mock_model, central_iteration=30)
        with pytest.raises(ValueError):
            cb.after_central_iteration(Metrics(),
                                       mock_model,
                                       central_iteration=31)


class TestWandbCallback:

    @patch('pfl.callback.WandbCallback.wandb')
    def test_callback(self, mock_wandb, mock_model, metrics):

        wandb_project_id = 'wandb-project'
        wandb_experiment_name = 'llm'
        wandb_config = {'cheat_code': 1}
        wandb_kwargs = {'param1': 1, 'param2': 2}
        cb = WandbCallback(wandb_project_id, wandb_experiment_name,
                           wandb_config, **wandb_kwargs)

        mock_wandb.init.assert_not_called()
        cb.on_train_begin(model=mock_model)
        mock_wandb.init.assert_called_once_with(project='wandb-project',
                                                name='llm',
                                                config={'cheat_code': 1},
                                                param1=1,
                                                param2=2)

        mock_wandb.log.assert_not_called()
        cb.after_central_iteration(metrics, mock_model, central_iteration=1)
        data = {
            'Val population | loss': 10.0,
            'Train population | number of data points': 2
        }
        mock_wandb.log.assert_called_once_with(data, step=1)

        mock_wandb.finish.assert_not_called()
        cb.on_train_end(model=mock_model)
        mock_wandb.finish.assert_called_once()
