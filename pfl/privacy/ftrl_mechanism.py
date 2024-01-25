# Copyright © 2023-2024 Apple Inc.
# Some of the code in this file is adapted from:
#
# google-research/federated:
# Copyright 2023, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
"""
Banded matrix factorization mechanism based on primal optimization algorithms.
Reference: https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py.  # pylint: disable=line-too-long
"""

import math
import os
import time
from typing import Callable, Generic, Optional, Tuple, TypeVar

import numpy as np

from pfl.exception import MatrixFactorizationError
from pfl.internal.bridge.factory import FrameworkBridgeFactory as bridges
from pfl.internal.ops import get_ops
from pfl.internal.platform.selector import get_platform
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from .gaussian_mechanism import GaussianMechanism, get_noise_stddev
from .privacy_accountant import PrivacyAccountant
from .privacy_snr import SNRMetric

Tensor = TypeVar('Tensor')


class FTRLMatrixFactorizer(Generic[Tensor]):
    """
    Class for factorizing matrices for matrix mechanism based on solving the
    optimization problem in Equation 6 in https://arxiv.org/pdf/2306.08153.pdf.

    :param workload_matrix:
        The input workload, n x n lower triangular matrix.
    :param mask:
        A boolean matrix describing the constraints on the gram matrix
        X = C^T C.
    """

    def __init__(self,
                 workload_matrix: np.ndarray,
                 mask: Optional[np.ndarray] = None):
        assert workload_matrix.shape == mask.shape
        if mask is None:
            mask = np.ones_like(workload_matrix, dtype=workload_matrix.dtype)
        self._n = workload_matrix.shape[1]
        self._A = get_ops().to_tensor(workload_matrix)
        # Mask determine which entries of X are allowed to be non-zero.
        self._mask = get_ops().to_tensor(mask)
        self._X_init = get_ops().to_tensor(
            np.eye(self._n, dtype=workload_matrix.dtype))

    def optimize(self, iters: int = 1000) -> Tensor:
        """
        Optimize the strategy matrix with an iterative gradient-based method.
        """
        X = self._X_init
        if not np.all(get_ops().to_numpy((1 - self._mask) * X == 0)):
            raise ValueError(
                'Initial X matrix is nonzero in indices i, j where '
                'i != j and some user can participate in rounds i and '
                'j. Such entries being zero is generally assumed by the '
                'optimization code here and downstream consumers in '
                'order to easily reason about sensitivity.')
        loss, dX = bridges.ftrl_bridge().loss_and_gradient(
            self._A, X, self._mask)
        prev_X = X  # X at previous iteration
        prev_dX = dX  # dX at previous iteration
        prev_loss = loss  # Loss at previous iteration
        Z = dX  # The negative search direction (different from dX in general)
        for _ in range(iters):
            step_size = 1.0
            for _ in range(30):
                X = prev_X - step_size * Z
                try:
                    loss, dX = bridges.ftrl_bridge().loss_and_gradient(
                        self._A, X, self._mask)
                except MatrixFactorizationError:
                    step_size *= 0.25
                    continue
                if loss < prev_loss:
                    prev_loss = loss
                    break
            if bridges.ftrl_bridge().terminate_fn(dX=dX):
                # Early-return triggered; return X immediately.
                return X
            Z = bridges.ftrl_bridge().lbfgs_direction(X, dX, prev_X, prev_dX)
            prev_X = X
            prev_dX = dX
        return X


class ForwardSubstitution(Generic[Tensor]):
    """
    Solve for X in LX = Y in an online manner using forward substitution where
    L is a lower-triangular matrix, as in Algorithm 9 in
    https://arxiv.org/pdf/2306.08153.pdf.

    :param matrix:
        The lower triangular matrix L.
    :param bandwidth:
        Optional bandwidth of L.
    """

    def __init__(self, matrix: np.ndarray, bandwidth: Optional[int] = None):
        self._matrix = get_ops().to_tensor(matrix)
        self._bandwidth = bandwidth
        self._previous_solved = None
        self._i = 0

    def step(self, y_i: Tensor) -> Tensor:
        r"""
        At step i, $X_i = (Y_i - \sum_{j=1}^{i-1} L_{i,j} X_j) / L_{i,i}$
        """
        y_i = get_ops().to_tensor(y_i)
        if self._previous_solved is not None:
            b, _ = get_ops().get_shape(self._previous_solved)
            start_i = 0 if self._bandwidth is None else self._i - b
            y_i = y_i - (self._matrix[self._i:self._i + 1, start_i:self._i]
                         @ self._previous_solved)[0]
        x_i = y_i / self._matrix[self._i, self._i]
        x_i_copy = get_ops().clone(x_i)[None]
        if self._previous_solved is None:
            self._previous_solved = x_i_copy
        else:
            self._previous_solved = get_ops().concatenate(
                [self._previous_solved, x_i_copy], axis=0)
        if self._bandwidth is not None and len(
                self._previous_solved) > self._bandwidth - 1:
            self._previous_solved = self._previous_solved[1:]
        self._i += 1
        return x_i


class BandedMatrixFactorizationMechanism(GaussianMechanism):
    r"""
    Banded matrix factorization mechanism https://arxiv.org/pdf/2306.08153.pdf.
    Matrix Mechanism in PFL privately estimates $AX = B(CX + Z)$, where
    $X \in R^{T\times d} = [x_1, x_2, \cdots x_T]$ is the series of aggregated
    gradients at each PFL iteration. $A$ is the workload matrix which is set
    to the lower triangular matrix with all ones. $Z$ is the Gaussian noise
    added. $BC = A$  is a factorization of $A$ such that the noise added is
    minimized.

    In the banded matrix setting, $C$ is a lower triangular banded matrix and
    the mechanism can be written as $A(X + C^{-1}Z)$, where in each step, a
    correlated noise from previous steps is added.

    :param clipping_bound:
        The norm bound for clipping.
    :param num_iterations:
        The number of times the mechanism will be applied.
    :param min_separation:
        Minimum number of iteration gap between two participation of a single
        device.
    :param make_privacy_accountant:
        Lambda function that takes number of compositions as input and returns
        privacy accountant, of type PrivacyAccountantKind.

    ::example
        .. code-block:: python

            make_privacy_accountant = lambda num_compositions:
                PLDPrivacyAccountant(num_compositions, **other_params)
            BandedMatrixFactorizationMechanism(clipping_bound, num_iterations,
                min_separation, make_privacy_accountant)
    """

    _QUERY_MATRIX_NPY_NAME = 'query_matrix.npy'
    _QUERY_MATRIX_DIR_PATH = 'banded_mf_dp_ftrl'

    def __init__(self, clipping_bound: float, num_iterations: int,
                 min_separation: int,
                 make_privacy_accountant: Callable[[int], PrivacyAccountant]):

        max_participation = num_iterations // min_separation + int(
            num_iterations % min_separation != 0)

        privacy_accountant = make_privacy_accountant(max_participation)

        super().__init__(
            clipping_bound=clipping_bound,
            relative_noise_stddev=privacy_accountant.cohort_noise_parameter)

        query_matrix_dir = get_platform().create_checkpoint_directories(
            [self._QUERY_MATRIX_DIR_PATH])[0]
        query_matrix_path = os.path.join(query_matrix_dir,
                                         self._QUERY_MATRIX_NPY_NAME)

        # Off-by-one from definition of banded matrix
        bandwidth = min_separation - 1
        banded_matrix_mask = self._get_banded_matrix_mask(
            num_iterations, bandwidth)

        if get_ops().distributed.local_rank == 0:
            # Run matrix factorization on 1 local process
            prefix_sum_matrix = np.tril(np.ones(num_iterations))
            # Optimize to get the gram matrix X = C^T C
            X = get_ops().to_numpy(
                FTRLMatrixFactorizer(prefix_sum_matrix,
                                     banded_matrix_mask).optimize())
            # Decompose X to get the query matrix C
            self._C = np.array(np.linalg.cholesky(X))
            np.save(query_matrix_path, self._C)

        while not os.path.exists(query_matrix_path):
            time.sleep(2)

        self._C = np.load(query_matrix_path)
        # Correlating noise from different iterations with C^{-1}
        self._C_inv_noise: ForwardSubstitution = ForwardSubstitution(
            self._C, bandwidth)

    @property
    def query_matrix(self) -> np.ndarray:
        return self._C

    @staticmethod
    def _get_banded_matrix_mask(n: int, bandwidth: int) -> np.ndarray:
        mask = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i - j) >= bandwidth:
                    mask[i, j] = 0
        return mask

    @staticmethod
    def _generate_flattened_noise(dimension: int, stddev: float, seed: int):
        noise_template = get_ops().to_tensor(np.zeros(dimension))
        return get_ops().add_gaussian_noise(tensors=[noise_template],
                                            stddev=stddev,
                                            seed=seed)[0]

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        noise_stddev = get_noise_stddev(self._clipping_bound,
                                        self._relative_noise_stddev)

        # Generate a flattened noise and correlate with noises from previous
        # iterations.
        num_dimensions = statistics.num_parameters
        flattened_noise = self._generate_flattened_noise(
            num_dimensions, noise_stddev, seed)
        correlated_noise = self._C_inv_noise.step(flattened_noise)

        # Reshape the flattened noise and added to delta
        _metadata, weights = statistics.get_weights()
        shapes = [get_ops().get_shape(w) for w in weights]
        reshaped_noise_weights = get_ops().reshape(correlated_noise, shapes)
        noise = statistics.from_weights(_metadata * 0, reshaped_noise_weights)
        data_with_noise = statistics + noise

        signal_norm = get_ops().global_norm(weights, order=2)
        noise_norm = get_ops().global_norm(reshaped_noise_weights, order=2)
        squared_error = noise_norm**2
        noise_stddev = math.sqrt(squared_error / num_dimensions)

        metrics = Metrics([(name_formatting_fn('DP noise std. dev.'),
                            Weighted.from_unweighted(noise_stddev)),
                           (name_formatting_fn('signal-to-DP-noise ratio'),
                            SNRMetric(signal_norm, squared_error))])

        return data_with_noise, metrics
