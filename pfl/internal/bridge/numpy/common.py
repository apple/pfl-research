# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import dill

from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.base import StatefulModel

from ..base import CommonFrameworkBridge


class NumpyCommonBridge(CommonFrameworkBridge[StatefulModel,
                                              NNTrainHyperParams]):

    @staticmethod
    def save_state(state: object, path: str):
        with open(path, 'wb') as f:
            dill.dump(state, f)

    @staticmethod
    def load_state(path: str):
        with open(path, 'rb') as f:
            state = dill.load(f)
        return state
