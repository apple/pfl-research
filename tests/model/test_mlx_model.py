# Copyright © 2023-2024 Apple Inc.
from unittest.mock import Mock

import numpy as np
import pytest

from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.internal.ops.common_ops import check_mlx_installed
from pfl.internal.ops.selector import _internal_reset_framework_module

if check_mlx_installed():
    # pylint: disable=ungrouped-imports
    import mlx
    _internal_reset_framework_module()


@pytest.mark.skipif(not check_mlx_installed(), reason='MLX not installed')
class TestMLXModel:
    """
    Contains all tests that are unique to MLXModel.
    """

    def test_save_and_load_central_optimizer_impl(
            self, mlx_model_setup, check_save_and_load_central_optimizer_impl):
        """
        Test if central optimizer could be save and restored
        """
        mlx_model_setup.model._central_optimizer = mlx.optimizers.Adam(  # pylint: disable=protected-access
            learning_rate=1.0)
        check_save_and_load_central_optimizer_impl(mlx_model_setup)
