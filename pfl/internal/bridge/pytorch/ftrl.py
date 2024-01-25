# Copyright © 2023-2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google-research/federated:
# Copyright 2022, Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").
"""
Primal optimization algorithms for multi-epoch matrix factorization.
Reference: https://github.com/google-research/federated/blob/master/multi_epoch_dp_matrix_factorization/multiple_participations/primal_optimization.py.  # pylint: disable=line-too-long
"""

from typing import Tuple

import torch

from pfl.exception import MatrixFactorizationError

from ..base import FTRLFrameworkBridge


class PyTorchFTRLBridge(FTRLFrameworkBridge[torch.Tensor]):

    @staticmethod
    @torch.no_grad()
    def loss_and_gradient(
            A: torch.Tensor, X: torch.Tensor,
            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        def _solve_positive_definite(A: torch.Tensor,
                                     B: torch.Tensor) -> torch.Tensor:
            """
            Solve X for AX = B where A is a positive definite matrix.
            """
            C = torch.linalg.cholesky(A.detach())
            return torch.cholesky_solve(B, C)

        def _project_update(dX: torch.Tensor) -> torch.Tensor:
            """
            Project dX so that:
            (1) The diagonal of X will be unchanged by setting diagonal of dX
            to 0 to ensure the sensitivity of the mechanism.
            (2) dX[i,j] is set to zero if mask[i,j] = 0.
            """
            dX.fill_diagonal_(0)
            return dX * mask

        try:
            H = _solve_positive_definite(X, A.T)
            loss, gradients = torch.trace(H @ A), _project_update(-H @ H.T)
            assert not torch.any(torch.isnan(loss)).item()
            assert not torch.any(torch.isnan(gradients)).item()
        except Exception as e:
            raise MatrixFactorizationError from e
        else:
            return loss, gradients

    @staticmethod
    @torch.no_grad()
    def lbfgs_direction(X: torch.Tensor, dX: torch.Tensor,
                        prev_X: torch.Tensor,
                        prev_dX: torch.Tensor) -> torch.Tensor:
        S = X - prev_X
        Y = dX - prev_dX
        rho = 1.0 / torch.sum(Y * S)
        alpha = rho * torch.sum(S * dX)
        gamma = torch.sum(S * Y) / torch.sum(Y**2)
        Z = gamma * (dX - rho * torch.sum(S * dX) * Y)
        beta = rho * torch.sum(Y * Z)
        Z = Z + S * (alpha - beta)
        return Z

    @staticmethod
    def terminate_fn(dX: torch.Tensor) -> bool:
        return torch.abs(dX).amax().item() <= 1e-3
