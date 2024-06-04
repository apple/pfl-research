
import os
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, Union

import joblib
import numpy as np

from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam.base import ModelHyperParams
from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.internal.ops.selector import set_framework_module
from pfl.metrics import Metrics
from pfl.model.base import Model
from pfl.stats import MappedVectorStatistics

Tensor = TypeVar('Tensor')
FrameworkModelType = TypeVar('FrameworkModelType')


@dataclass(frozen=True)
class MDMModelHyperParams(ModelHyperParams):
    """
    Parameters for Polya-Mixture model.
    """
    num_components: int
    num_categories: int

    def __post_init__(self):
        if self.num_components is not None:
            assert self.num_components >= 1, (
                'Must have >= 1 component in Polya-Mixture model')
        if self.num_categories is not None:
            assert self.num_categories >= 2, (
                'Must have >= 2 categories being modelled with Polya-Mixture')


MDMModelHyperParamsType = TypeVar('MDMModelHyperParamsType',
                                  bound=MDMModelHyperParams)


class MDMModel(Model, Generic[MDMModelHyperParamsType, Tensor]):
    """
    Polya Mixture model.

    Used Fixed-Point solver.

    Model that applies a weighted version of the fixed point Polya update from
    https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf to the alpha
    of each mixture component.
    :param phi:
        np.ndarray of shape (number_mixture_components,) giving the weight of
        each component, sums to 1.
    :param alphas:
        np.ndarray of shape (number_mixture_components, num_categories), stores
        the alpha parameter for the Dirichlet of each mixture component.
    """
    set_framework_module(pytorch_ops)
    _MODEL_CKPT_NAME = "polya-mixture.joblib"

    def __init__(self,
                 alphas: Optional[Union[List, np.ndarray]] = None,
                 phi: Optional[Union[List, np.ndarray]] = None,
                 num_samples_distribution: Optional[Union[List,
                                                          np.ndarray]] = None):

        self._alphas = get_ops().to_tensor(
            alphas) if alphas is not None else None
        self._phi = get_ops().to_tensor(phi) if phi is not None else None
        self._num_samples_distribution = get_ops().to_tensor(
            num_samples_distribution
        ) if num_samples_distribution is not None else None

    def _to_dict(self):
        return {
            'alphas': self._alphas,
            'phi': self._phi,
            'num_samples_distribution': self._num_samples_distribution
        }

    def save(self, dir_path: str) -> None:
        """
        Save a Polya-Mixture model to disk.

        :param dir_path:
            Path to which to save Polya-Mixture model will be saved.
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        save_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
        joblib.dump(self._to_dict(), save_path)

    def load(self, dir_path: str) -> None:
        save_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
        if not os.path.exists(save_path):
            raise CheckpointNotFoundError(save_path)

        parameters = joblib.load(save_path)
        try:
            self._alphas = parameters['alphas']
            self._phi = parameters['phi']
            self._num_samples_distribution = parameters[
                'num_samples_distribution']
        except KeyError as e:
            raise KeyError(
                'Polya-Mixture model checkpoint does not '
                'contain required keys: "alpha", "phi", '
                f'"num_components", "num_categories", "num_samples_distribution": {e}'
            )

    @property
    def phi(self) -> Union[np.ndarray, List[float]]:
        return self._phi

    @property
    def alphas(self) -> Union[np.ndarray, List[float]]:
        return self._alphas

    @property
    def num_samples_distribution(self) -> Union[np.ndarray, List[float]]:
        return self._num_samples_distribution

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['MDMModel', Metrics]:

        self._alphas = statistics['alphas']
        self._phi = statistics['phi']
        self._num_samples_distribution = statistics['num_samples_distribution']

        return self, Metrics()


MDMModelType = TypeVar('MDMModelType', bound=MDMModel)
