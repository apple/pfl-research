# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/federated:
# Copyright 2023, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
"""
Primal optimization algorithms for multi-epoch matrix factorization.
Reference: https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py.  # pylint: disable=line-too-long
"""

from typing import Tuple

import tensorflow as tf

from pfl.exception import MatrixFactorizationError
from pfl.internal.ops.tensorflow_ops import tf_function
from ..base import FTRLFrameworkBridge


class TFFTRLBridge(FTRLFrameworkBridge[tf.Tensor]):

    @staticmethod
    def loss_and_gradient(A: tf.Tensor, X: tf.Tensor,
                          mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:

        @tf_function
        def _solve_positive_definite(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
            """
            Solve X for AX = B where A is a positive definite matrix.
            """
            C = tf.linalg.cholesky(tf.stop_gradient(A))
            return tf.linalg.cholesky_solve(C, B)

        @tf_function
        def _project_update(dX: tf.Tensor) -> tf.Tensor:
            """
            Project dX so that:
            (1) The diagonal of X will be unchanged by setting diagonal of dX
            to 0 to ensure the sensitivity of the mechanism.
            (2) dX[i,j] is set to zero if mask[i,j] = 0.
            """
            dX = tf.linalg.set_diag(dX, tf.zeros(dX.shape[0], dtype=dX.dtype))
            return dX * mask

        @tf_function
        def _loss_and_gradient_tf(A: tf.Tensor,
                                  X: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            H = _solve_positive_definite(X, tf.transpose(A))
            return tf.linalg.trace(H @ A), _project_update(
                -H @ tf.transpose(H))

        try:
            loss, gradient = _loss_and_gradient_tf(A, X)
            assert not bool(tf.reduce_any(tf.math.is_nan(loss)).numpy())
            assert not bool(tf.reduce_any(tf.math.is_nan(gradient)).numpy())
            return loss, gradient
        except Exception as e:
            raise MatrixFactorizationError from e

    @staticmethod
    @tf_function
    def lbfgs_direction(X: tf.Tensor, dX: tf.Tensor, prev_X: tf.Tensor,
                        prev_dX: tf.Tensor) -> tf.Tensor:
        S = X - prev_X
        Y = dX - prev_dX
        rho = 1.0 / tf.reduce_sum(Y * S)
        alpha = rho * tf.reduce_sum(S * dX)
        gamma = tf.reduce_sum(S * Y) / tf.reduce_sum(Y**2)
        Z = gamma * (dX - rho * tf.reduce_sum(S * dX) * Y)
        beta = rho * tf.reduce_sum(Y * Z)
        Z = Z + S * (alpha - beta)
        return Z

    @staticmethod
    @tf_function
    def terminate_fn(dX: tf.Tensor) -> bool:
        return float(tf.reduce_max(tf.abs(dX))) <= 1e-3
