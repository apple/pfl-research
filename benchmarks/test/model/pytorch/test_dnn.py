# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from typing import Tuple

import numpy as np
import pytest

from pfl.internal.ops import get_pytorch_major_version

if get_pytorch_major_version():
    from model.pytorch import simple_dnn as pytorch_dnn
    import torch


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
class TestPyTorchDNN:

    @pytest.mark.parametrize('input_shape', [(32, 32, 3), (28, 28, 1)])
    @pytest.mark.parametrize('output_shape', [10, 100])
    def test_output_shape(self, input_shape: Tuple[int, ...],
                          output_shape: int):
        pytorch_model = pytorch_dnn(input_shape, output_shape)
        inputs = np.random.normal(size=(1, ) + input_shape).astype(np.float32)
        logits = pytorch_model(torch.from_numpy(inputs))
        assert int(logits.shape[-1]) == output_shape

    @pytest.mark.parametrize('input_shape', [(32, 32, 3), (28, 28, 1)])
    @pytest.mark.parametrize('output_shape', [10, 100])
    def test_num_parameters(self, input_shape: Tuple[int, ...],
                            output_shape: int):
        pytorch_model = pytorch_dnn(input_shape, output_shape)
        # 1st dense layer, 200 units.
        num_parameters = np.prod(input_shape) * 200 + 200
        # 2st dense layer, 200 units.
        num_parameters += 200 * 200 + 200
        # output layer.
        num_parameters += 200 * output_shape + output_shape
        pytorch_model_num_parameters = sum(
            [np.prod(tuple(var.shape)) for var in pytorch_model.parameters()])
        assert num_parameters == pytorch_model_num_parameters
