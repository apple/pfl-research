# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
# Some of the code in this file is adapted from:
#
# google-research/federated:
# Copyright 2023, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.internal.bridge.factory import FrameworkBridgeFactory as bridges
from pfl.internal.ops import get_ops, get_pytorch_major_version, get_tf_major_version
from pfl.privacy import PLDPrivacyAccountant
from pfl.privacy.ftrl_mechanism import BandedMatrixFactorizationMechanism, ForwardSubstitution, FTRLMatrixFactorizer
from pfl.stats import MappedVectorStatistics

try:
    import jax.numpy as jnp  # type: ignore
    import jax.scipy as jsp  # type: ignore
    from jax.config import config  # type: ignore

    config.update('jax_enable_x64', True)
    jax_installed = True
except ModuleNotFoundError:
    jax_installed = False

# These fixtures set the internal framework module.
framework_fixtures = [
    pytest.param(
        lazy_fixture('tensorflow_ops'),  # type: ignore
        marks=[
            pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')
        ]),
    pytest.param(
        lazy_fixture('pytorch_ops'),  # type: ignore
        marks=[
            pytest.mark.skipif(not get_pytorch_major_version(),
                               reason='PyTorch not installed')
        ]),
]


def get_banded_mask(n, bandwidth):
    mask = np.ones((n, n))
    if bandwidth is not None:
        for i in range(n):
            for j in range(n):
                if abs(i - j) >= bandwidth:
                    mask[i, j] = 0
    return mask


@pytest.mark.parametrize('n', [8, 16])
@pytest.mark.parametrize('d', [8, 16])
@pytest.mark.parametrize('bandwidth', [None, 3, 6])
@pytest.mark.parametrize('ops_module', framework_fixtures)
def test_forward_substitution(n, d, bandwidth, ops_module):
    mask = get_banded_mask(n, bandwidth)
    # https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
    M = np.random.rand(n, n)
    M_abs_sum = np.sum(np.abs(M), axis=1)
    np.fill_diagonal(M, M_abs_sum)
    C = M * np.tri(n) * mask
    C_inv = np.linalg.inv(C)
    Z = np.random.rand(n, d)
    expected_C_inv_Z = C_inv @ Z
    forward_substitution = ForwardSubstitution(C, bandwidth)
    for i in range(n):
        C_inv_Z_i = get_ops().to_numpy(forward_substitution.step(Z[i]))
        np.testing.assert_allclose(expected_C_inv_Z[i],
                                   C_inv_Z_i,
                                   atol=1e-6,
                                   rtol=1e-6)


@pytest.mark.parametrize('n', [8, 16])
@pytest.mark.parametrize('bandwidth', [4, 6])
@pytest.mark.parametrize('ops_module', framework_fixtures)
class TestMatrixFactorization:

    @staticmethod
    def check_symmetric(X):
        """Checks if a given matrix is symmetric."""
        assert np.allclose(X, X.T)

    @staticmethod
    def check_pd(X):
        """Checks if a given matrix is positive definite."""
        assert np.all(np.linalg.eigvals(X) > 0)

    @staticmethod
    def check_banded(X, bandwidth):
        """Checks if a given matrix is banded."""
        n = X.shape[0]
        for i in range(n):
            for j in range(n):
                assert not (abs(i - j) > bandwidth and X[i, j] != 0)

    @staticmethod
    def check_equal_norm(C):
        """Checks that column norms of matrix are equal."""
        norms = np.linalg.norm(C, axis=1)
        assert max(norms) - min(norms) <= 1e-7

    @staticmethod
    def check_improved(A, X, X_init, mask):
        loss_init = bridges.ftrl_bridge().loss_and_gradient(A, X_init, mask)[0]
        loss_optimized = bridges.ftrl_bridge().loss_and_gradient(A, X, mask)[0]
        assert loss_optimized <= loss_init

    @staticmethod
    def check_optimal(X, opt):
        _, projected_grad = opt.loss_and_gradient(X)
        assert np.abs(projected_grad).max() < 1e-3

    def test_optimization(self, n, bandwidth, ops_module):
        A = np.tri(n)
        mask = get_banded_mask(n, bandwidth)
        X = FTRLMatrixFactorizer(A, mask).optimize()
        self.check_improved(ops_module.to_tensor(A), X,
                            ops_module.to_tensor(np.eye(n)),
                            ops_module.to_tensor(mask))
        X = ops_module.to_numpy(X)
        self.check_symmetric(X)
        self.check_pd(X)
        self.check_banded(X, bandwidth)
        C = np.linalg.cholesky(X)
        self.check_equal_norm(C)

    @staticmethod
    def jax_matrix_factorization(A, mask):
        # Adapted from Google's original implementation
        # https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py
        n = len(A)
        A = jnp.asarray(A)

        def project_update(dX):
            dX = dX.at[jnp.diag_indices(n)].set(0)
            return dX * mask

        def loss_and_gradient(X):
            H = jsp.linalg.solve(X, A.T, assume_a='pos')
            return jnp.trace(H @ A), project_update(-H @ H.T)

        def lbfgs_direction(X, dX, X1, dX1):
            S = X - X1
            Y = dX - dX1
            rho = 1.0 / jnp.sum(Y * S)
            alpha = rho * jnp.sum(S * dX)
            gamma = jnp.sum(S * Y) / jnp.sum(Y**2)
            Z = gamma * (dX - rho * jnp.sum(S * dX) * Y)
            beta = rho * jnp.sum(Y * Z)
            Z = Z + S * (alpha - beta)
            return Z

        X = jnp.eye(n)
        loss, dX = loss_and_gradient(X)
        X1 = X  # X at previous iteration
        dX1 = dX  # dX at previous iteration
        loss1 = loss  # Loss at previous iteration
        Z = dX  # The negative search direction (different from dX in general)
        for _ in range(1000):
            step_size = 1.0
            for _ in range(30):
                X = X1 - step_size * Z
                loss, dX = loss_and_gradient(X)
                if jnp.isnan(loss).any() or jnp.isnan(dX).any():
                    step_size *= 0.25
                elif loss < loss1:
                    loss1 = loss
                    break
            if jnp.abs(dX).max() <= 1e-3:
                return X
            Z = lbfgs_direction(X, dX, X1, dX1)
            X1 = X
            dX1 = dX
        return np.asarray(X)

    @pytest.mark.skipif(not jax_installed, reason="JAX not installed")
    def test_optimization_against_jax(self, n, bandwidth, ops_module):
        A = np.tri(n)
        mask = get_banded_mask(n, bandwidth)
        X = FTRLMatrixFactorizer(A, mask).optimize()
        X = ops_module.to_numpy(X)
        jax_X = self.jax_matrix_factorization(A, mask)
        np.testing.assert_allclose(X, jax_X, atol=1e-6, rtol=1e-6)


@pytest.fixture(autouse=True)
def mock_artifacts_dir_creation(tmp_path):

    def _checkpoint_dir_side_effect(args):
        dirs = []
        for dir_name in args:
            posix_dir = (tmp_path / dir_name).as_posix()
            os.makedirs(posix_dir, exist_ok=True)
            dirs.append(posix_dir)
        return dirs

    with patch('pfl.privacy.ftrl_mechanism.get_platform') as mock_get_platform:
        mock_platform = MagicMock(
            create_checkpoint_directories=_checkpoint_dir_side_effect)
        mock_get_platform.return_value = mock_platform

        yield


@pytest.mark.parametrize('ops_module', [
    pytest.param(lazy_fixture('tensorflow_ops'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ]),
    pytest.param(lazy_fixture('pytorch_ops'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ]),
])
class TestBandedMatrixFactorizationMechanism:

    @staticmethod
    def _get_banded_matrix_mechanism():
        make_privacy_accountant = lambda num_compositions: MagicMock(
            cohort_noise_parameter=1.1713266372680664)
        return BandedMatrixFactorizationMechanism(
            clipping_bound=1.0,
            num_iterations=16,
            min_separation=4,
            make_privacy_accountant=make_privacy_accountant)

    def test_save_query_matrix(self, ops_module, tmp_path):
        mechanism = self._get_banded_matrix_mechanism()
        save_path = os.path.join(
            tmp_path,
            mechanism._QUERY_MATRIX_DIR_PATH,  # pylint: disable=protected-access
            mechanism._QUERY_MATRIX_NPY_NAME)  # pylint: disable=protected-access
        assert os.path.exists(save_path)

    def test_correlated_noise(self, ops_module):
        mechanism = self._get_banded_matrix_mechanism()
        C_inv = np.linalg.inv(mechanism.query_matrix)
        d = 10
        noise_history = []
        for i in range(16):
            zeros_stats = MappedVectorStatistics(
                {'': ops_module.to_tensor(np.zeros(d))})
            C_inv_i = C_inv[i]
            correlated_noise_i, _ = mechanism.add_noise(zeros_stats,
                                                        cohort_size=1,
                                                        seed=i + 1)
            correlated_noise_i = correlated_noise_i['']
            expected_noise_i = ops_module.add_gaussian_noise(
                tensors=[ops_module.to_tensor(np.zeros(d))],
                stddev=mechanism.relative_noise_stddev,
                seed=i + 1)[0]
            noise_history.append(expected_noise_i)
            expected_correlated_noise_i = 0
            for c, noise in zip(C_inv_i[:i + 1], noise_history):
                expected_correlated_noise_i += c * noise
            np.testing.assert_allclose(
                ops_module.to_numpy(expected_correlated_noise_i),
                ops_module.to_numpy(correlated_noise_i),
                atol=1e-6,
                rtol=1e-6)
