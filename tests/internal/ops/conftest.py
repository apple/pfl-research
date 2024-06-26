# Copyright © 2023-2024 Apple Inc.

import contextlib
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import pytest

# Struct to return from the ops setup fixtures.
OpsSetup = namedtuple(
    'OpsSetup',
    [
        # The ops module.
        'ops',
        # A list of tensors of the correct type for the ops module. This
        # might be NumPy arrays if the selected Deep Learning framework
        # does not support tensors end-to-end.
        'ops_variables',
        # A variable of the Deep Learning framework.
        'variable',
        # A reference variable of the Deep Learning framework
        'reference',
        # Context manager for mocking any noise operations to return 1.
        'deterministic_noise',
        # A helper function to convert a variable into a `numpy.ndarray`.
        'variable_to_numpy_func',
        # A helper function to convert a `numpy.ndarray` into a tensor.
        'numpy_to_tensor_func'
    ])


@pytest.fixture
def check_equal_tensors():

    def _check_equal_tensors(numpy_vars,
                             tensors,
                             ops_setup,
                             almost_equal=False):
        for ndarray, tensor in zip(numpy_vars, tensors):
            if almost_equal:
                np.testing.assert_array_almost_equal(
                    ndarray, ops_setup.ops.to_numpy(tensor))
            else:
                np.testing.assert_array_equal(ndarray,
                                              ops_setup.ops.to_numpy(tensor))

    return _check_equal_tensors


@pytest.fixture
def numpy_vars():
    return [
        np.arange(4).reshape((2, 2)).astype(np.float32),
        np.array([0], dtype=np.float32)
    ]


@pytest.fixture(scope='function')
def tensorflow_ops_setup(tensorflow_ops, numpy_vars):
    import tensorflow as tf  # type: ignore

    @contextlib.contextmanager
    def deterministic_noise():
        with patch('pfl.internal.ops.tensorflow_ops._normal_dist',
                   autospec=True) as mock_normal, patch(
                       'pfl.internal.ops.tensorflow_ops._laplace_dist',
                       autospec=True) as mock_laplace:
            mock_normal.sample.side_effect = lambda t, seed: tf.ones_like(t)
            mock_laplace.sample.side_effect = lambda t, seed: tf.ones_like(t)
            yield

    yield OpsSetup(ops=tensorflow_ops,
                   ops_variables=[tf.convert_to_tensor(v) for v in numpy_vars],
                   variable=tf.Variable(np.zeros((3, 2), dtype=np.float32)),
                   reference=tf.Variable(np.ones((3, 2), dtype=np.float32)),
                   deterministic_noise=deterministic_noise,
                   variable_to_numpy_func=lambda v: v.numpy(),
                   numpy_to_tensor_func=lambda n: tf.convert_to_tensor(n))


@pytest.fixture(scope='function')
def pytorch_ops_setup(pytorch_ops, numpy_vars):
    import torch  # type: ignore

    @contextlib.contextmanager
    def deterministic_noise():
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        with patch('pfl.internal.ops.pytorch_ops.torch.Tensor.normal_',
                   autospec=True) as mock_normal, patch(
                       'pfl.internal.ops.pytorch_ops.torch.'
                       'distributions.Laplace.sample') as mock_laplace:
            mock_normal.side_effect = (
                lambda mean, std, generator: torch.tensor([1.], device=device))
            mock_laplace.side_effect = (
                lambda sample_shape: torch.tensor([1.], device=device))
            yield

    return OpsSetup(
        ops=pytorch_ops,
        ops_variables=[torch.tensor(v).cpu() for v in numpy_vars],
        variable=torch.Tensor(np.zeros((3, 2), dtype=np.float32)).cpu(),
        reference=torch.Tensor(np.ones((3, 2), dtype=np.float32)).cpu(),
        deterministic_noise=deterministic_noise,
        variable_to_numpy_func=lambda v: v.cpu().detach().numpy().copy(),
        numpy_to_tensor_func=lambda n: torch.Tensor(n).cpu())


@pytest.fixture(scope='function')
def numpy_ops_setup(numpy_ops, numpy_vars):

    @contextlib.contextmanager
    def deterministic_noise():
        with patch(
                'pfl.internal.ops.numpy_ops.np.random.normal',
                autospec=True) as mock_normal, \
             patch(
                'pfl.internal.ops.numpy_ops.np.random.laplace',
                autospec=True) as mock_laplace:
            mock_normal.side_effect = lambda loc, scale, size: np.ones(size)
            mock_laplace.side_effect = lambda loc, scale, size: np.ones(size)
            yield

    return OpsSetup(ops=numpy_ops,
                    ops_variables=numpy_vars,
                    variable=np.zeros((3, 2), dtype=np.float32),
                    reference=np.ones((3, 2), dtype=np.float32),
                    deterministic_noise=deterministic_noise,
                    variable_to_numpy_func=lambda v: v,
                    numpy_to_tensor_func=lambda n: np.asarray(n))


@pytest.fixture(scope='function')
def mlx_ops_setup(mlx_ops, numpy_vars):
    import mlx.core as mx

    @contextlib.contextmanager
    def deterministic_noise():
        with patch(
                'pfl.internal.ops.mlx_ops.mx.random.normal',
                autospec=True) as mock_normal, \
             patch(
                'pfl.internal.ops.mlx_ops.mx.random.uniform',
                autospec=True) as mock_uniform, \
             patch(
                'pfl.internal.ops.mlx_ops.mx.abs',
                autospec=True) as mock_abs:

            mock_normal.side_effect = lambda loc, scale, shape, dtype, key: mx.ones(
                shape)
            mock_uniform.side_effect = lambda low, high, shape, dtype, key: mx.ones(
                shape)
            # There is no value that satisfies log1p(-abs(u))==1, so we mock mx.abs instead
            mock_abs.side_effect = lambda val: mx.ones_like(val) - mx.e
            yield

    return OpsSetup(ops=mlx_ops,
                    ops_variables=[mx.array(v) for v in numpy_vars],
                    variable=mx.zeros((3, 2), dtype=mx.float32),
                    reference=mx.ones((3, 2), dtype=mx.float32),
                    deterministic_noise=deterministic_noise,
                    variable_to_numpy_func=lambda v: v,
                    numpy_to_tensor_func=lambda n: mx.array(n))
