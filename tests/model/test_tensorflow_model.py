# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest

from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops import get_tf_major_version
from pfl.internal.ops.selector import _internal_reset_framework_module

if get_tf_major_version() == 2:
    _internal_reset_framework_module()
else:
    mock_converter = None


def get_keras_optimizer_or_none():
    """
    Even if a test class is skipped, the body of the test class
    will still be parsed.
    This means that if the tests are skipped because TF is not installed pytest
    will crash because ``tf.keras.optimizers.SGD`` can't be found, which is
    in the parametrization.
    """
    if get_tf_major_version() > 0:
        import tensorflow as tf  # type: ignore
        return tf.keras.optimizers.SGD(1.0)
    else:
        return None


def get_keras_adam_optimizer_or_none():
    """
    Even if a test class is skipped, the body of the test class
    will still be parsed.
    This means that if the tests are skipped because TF version < 0 pytest will
    crash because ``tf.keras.optimizer.Adam`` can't be found, which is
    in the parametrization.
    """
    if get_tf_major_version() > 0:
        import tensorflow as tf  # type: ignore
        return tf.keras.optimizers.Adam()
    else:
        return None


def _get_meta_gradient_test_case(x, y, batch_size, num_epochs, lr):
    batch_size = len(x) if batch_size is None else batch_size
    w = np.array([[2., 4.], [3., 5.]])
    w_history = [w]

    for _ in range(num_epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Inner-loop forwardprop
            yhat = np.dot(x_batch, w_history[-1])

            # Inner-loop backprop
            g_loss = np.sign(yhat - y_batch)
            gradient = np.dot(x_batch.T, g_loss) / len(x_batch)
            w_history.append(w_history[-1] - lr * gradient)

    # For simplicity, use train data for the meta-gradient.
    x_test, y_test = x, y

    # test set forward prop
    y_pred = np.dot(x_test, w_history[-1])

    if num_epochs > 0:
        accumulated_hessians = np.ones_like(w)
        # Accumulate hessians for intermediate models in the training procedure.
        for w_intermediate in w_history[:-1]:

            # The derivative of sign is a unit impulse function.
            delta = np.zeros_like(y_test)
            pred_diff = np.dot(x_test, w_intermediate) - y_test
            delta[(pred_diff) == 0] = 1

            h = np.dot(x_test.T, delta)
            accumulated_hessians *= (np.eye(len(w)) - lr * h)

        meta_g_loss = np.sign(y_pred - y_test)
        # Meta gradient is gradient of final model multiplied by hessian for
        # model states in intermediate steps.
        meta_gradient = np.dot(np.dot(x_test.T, meta_g_loss),
                               accumulated_hessians) / len(x)
    else:
        # Calculate gradient without any training occured.
        meta_g_loss = np.sign(y_pred - y_test)
        meta_gradient = np.dot(x_test.T, meta_g_loss) / len(x)

    return meta_gradient


@pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
class TestTFModel:
    """
    Contains all tests that are unique to TFModel.
    """

    @pytest.mark.parametrize(
        'tensorflow_model_setup',
        ({
            'get_central_optimizer': get_keras_optimizer_or_none
        }, ),
        indirect=True)
    def test_apply_model_update_momentum(self, tensorflow_model_setup,
                                         check_apply_model_update_impl):
        """
        Test a different global optimizer (MomentumOptimizer) with tensorflow.
        With a momentum of 0, the result should be the same as SGD.
        """
        check_apply_model_update_impl(tensorflow_model_setup)

    @pytest.mark.parametrize('tensorflow_model_setup', ({
        'model_kwargs': {
            'checkpoint_format_hdf5': True
        }
    }, ),
                             indirect=True)
    def test_save_model_hdf5(self, tensorflow_model_setup):
        """ Test if model can be stored in HDF5 format """
        tensorflow_model_setup.model.save(
            str(tensorflow_model_setup.save_model_path))

        expected_checkpoint_file = 'model.h5'
        actual_checkpoint_files = [
            f.name for f in tensorflow_model_setup.load_model_path.iterdir()
        ]

        assert expected_checkpoint_file in actual_checkpoint_files

    @pytest.mark.parametrize('tensorflow_model_setup', ({
        'model_kwargs': {
            'checkpoint_format_hdf5': True
        }
    }, ),
                             indirect=True)
    def test_load_model_hdf5(self, tensorflow_model_setup,
                             check_save_and_load_model_impl):
        """ Test if model can be restored from hdf5 checkpoint. """
        check_save_and_load_model_impl(tensorflow_model_setup,
                                       tensorflow_model_setup.save_model_path)

    @pytest.mark.parametrize(
        'tensorflow_model_setup',
        ({
            'get_central_optimizer': get_keras_adam_optimizer_or_none
        }, ),
        indirect=True)
    def test_save_and_load_central_optimizer_impl(
            self, tensorflow_model_setup,
            check_save_and_load_central_optimizer_impl):
        """
        Test if central optimizer could be save and restored
        """
        check_save_and_load_central_optimizer_impl(tensorflow_model_setup)

    def test_local_train_metadata(self, tensorflow_model_setup, user_dataset):
        model = tensorflow_model_setup.model
        from pfl.internal.bridge.tensorflow.sgd import _make_train_step
        step_fn = _make_train_step(model)
        train_metadata = model.do_multiple_epochs_of(
            user_dataset,
            NNTrainHyperParams(local_learning_rate=0.1,
                               local_num_epochs=3,
                               local_batch_size=1),
            step_fn,
            max_grad_norm=None)
        assert train_metadata.num_steps == 6
