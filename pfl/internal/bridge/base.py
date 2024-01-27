# Copyright © 2023-2024 Apple Inc.
"""
This module contains abstract interfaces that act as
bridges between framework-agnostic code and specific
deep learning frameworks.

Extending ``CommonFrameworkBridge`` is required at a minimum for simulations
to support a deep learning framework. Additional bridges can be
implemented to support additional algorithms and deep learning frameworks.

This is similar to ``pfl.internal.ops``, but distinct for these reasons:

* Ops module is the lowest module of primitive framework-specific code that can
  be injected anywhere in the code. It does not depend on any other modules.
* This module works similar to ops module, but can have dependencies on data
  structures like Statistics, Metrics, Dataset, Model.
  Since ops module does not have any pfl components as dependencies, it
  can be used in the implementation of the data structures mentioned, while
  this module of bridges is restricted to higher-level components which it
  does not self-reference, e.g. Algorithm and Privacy mechanism.
"""
from typing import Protocol, Tuple, TypeVar

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParams
from pfl.model.base import StatefulModel
from pfl.stats import MappedVectorStatistics

StatefulModelType_contra = TypeVar('StatefulModelType_contra',
                                   bound=StatefulModel,
                                   contravariant=True)
ModelHyperParamsType_contra = TypeVar('ModelHyperParamsType_contra',
                                      bound=ModelHyperParams,
                                      contravariant=True)
Tensor = TypeVar('Tensor')


# These mypy errors are ignored because they will disappear once
# CommonFrameworkBridge starts using the two generic types in any
# of the interfaces:
# Contravariant type variable "StatefulModelType_contra" used in protocol
# where covariant one is expected  [misc]
# Contravariant type variable "ModelHyperParamsType_contra" used in protocol
# where covariant one is expected  [misc]
class CommonFrameworkBridge(Protocol[  # type: ignore[misc]
        StatefulModelType_contra, ModelHyperParamsType_contra]):
    """
    Functions that need framework-specific implementations and
    that are required for rudimentary support of that deep learning framework
    in pfl.
    All other bridges than this interface are optional for supporting
    certain algorithms.
    """

    @staticmethod
    def save_state(state: object, path: str):
        """
        Save ``state`` to disk at path ``path``.
        """

    @staticmethod
    def load_state(path: str):
        """
        Load state from disk at path ``path``.
        """


class SGDFrameworkBridge(Protocol[StatefulModelType_contra,
                                  ModelHyperParamsType_contra]):
    """
    Interface for functions that manipulate the model using
    stochastic gradient descent and need framework-specific
    implementations.
    """

    @staticmethod
    def do_sgd(model: StatefulModelType_contra,
               user_dataset: AbstractDatasetType,
               train_params: ModelHyperParamsType_contra) -> None:
        """
        Do multiple epochs of SGD with the given input data.

        :param model:
            The model to train.
        :param user_dataset:
            Dataset of type ``Dataset`` to train on.
        :param train_params:
            An instance of :class:`~pfl.hyperparam.base.ModelHyperParams`
            containing configuration for training.
        """
        pass


class FedProxFrameworkBridge(Protocol[StatefulModelType_contra,
                                      ModelHyperParamsType_contra]):
    """
    Interface for implementing the FedProx algorithm, by
    T. Li. et al. - Federated Optimization in Heterogeneous Networks
    (https://arxiv.org/pdf/1812.06127.pdf),
    for a particular Deep Learning framework.
    """

    @staticmethod
    def do_proximal_sgd(model: StatefulModelType_contra,
                        user_dataset: AbstractDatasetType,
                        train_params: ModelHyperParamsType_contra,
                        mu: float) -> None:
        """
        Do multiple local epochs of SGD with the FedProx
        proximal term added to the loss (Equation 2)
        """
        pass


class SCAFFOLDFrameworkBridge(Protocol[StatefulModelType_contra,
                                       ModelHyperParamsType_contra]):
    """
    Interface for implementing the SCAFFOLD algorithm, by
    S. P. Karimireddy et al. - SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning.
    (https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf),
    for a particular Deep Learning framework.
    """

    @staticmethod
    def do_control_variate_sgd(
        model: StatefulModelType_contra,
        user_dataset: AbstractDatasetType,
        train_params: ModelHyperParamsType_contra,
        local_c: MappedVectorStatistics,
        server_c: MappedVectorStatistics,
    ) -> None:
        """
        Do multiple local epochs of SGD with local control
        variate ($c_i$) and server control variate ($c$),
        see Algorithm 1.
        """
        pass


class FTRLFrameworkBridge(Protocol[Tensor]):
    """
    Interface for implementing factorizing banded matrix for FTRL mechanism, by
    Choquette-Choo et al. - (Amplified) Banded Matrix Factorization: A unified
    approach to private training (https://arxiv.org/pdf/2306.08153.pdf),
    for a particular deep Learning framework.
    """

    @staticmethod
    def loss_and_gradient(A: Tensor, X: Tensor,
                          mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the loss $\tr[A^T A X^{-1}]$ and the associated gradient
        $dX = -X^{-1} A^T A X^{-1}$ from the optimization problem in Equation 6
        in https://arxiv.org/pdf/2306.08153.pdf.
        """
        pass

    @staticmethod
    def lbfgs_direction(X: Tensor, dX: Tensor, prev_X: Tensor,
                        prev_dX: Tensor) -> Tensor:
        """
        Given the current/previous iterates (X and X1) and the current/previous
        gradients (dX and dX1), compute a search direction (Z) according to the
        LBFGS update rule.
        """

    @staticmethod
    def terminate_fn(dX: Tensor) -> bool:
        """
        Criterion to terminate optimization based on dX.
        """
