# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import numpy as np
import torch  # type: ignore
import torch.nn as nn
import torch.nn.functional as F
from pfl.metrics import Weighted
import types
from typing import Tuple, List

from .layer import Transpose2D
from .metrics import image_classification_metrics, image_classification_loss
from ..numpy.metrics import AveragedPrecision, MacroWeighted


def multi_label_cnn(
    model_type: str,
    num_outputs: int,
    channel_mean: List[float],
    channel_stddevs: List[float],
    pretrained: bool,
):
    """
    A CNN used for multi-label classification task.

    :param model_type:
        The architecture of the model.
    :param num_outputs:
        Size of the output multi-label classification layer.
    :param channel_mean:
        Means for input image RGB channels.
    :param channel_stddevs:
        Standard deviations for input image RGB channels.
    :param pretrained:
        Whether to use ImageNet pretrained model.

    :return:
        A Pytorch CNN module for multi-label classification.
    """

    import torchvision.models  # type: ignore
    import torchvision.transforms as transforms  # type: ignore
    from .module_modification import (validate_no_batchnorm,
                                      freeze_batchnorm_modules,
                                      convert_batchnorm_modules)

    torchvision_models = torchvision.models.__dict__

    class MultiLabelCNN(nn.Module):
        """
        Wrapper of torchvision.models used for PFL training on multi-label
        classification task, e.g. on FLAIR dataset.
        """

        def __init__(
            self,
            torchvision_model_type: str,
            num_outputs: int,
            channel_mean: List[float],
            channel_stddevs: List[float],
            pretrained: bool,
        ):
            super().__init__()
            self._num_outputs = num_outputs

            # input image transformation, same as standard ImageNet training
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(channel_mean, channel_stddevs)
            ])
            self.eval_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Normalize(channel_mean, channel_stddevs)
            ])

            # per-class binary cross-entropy for multi-label classification
            # learning objective
            self.loss_fct = nn.BCEWithLogitsLoss()

            # https://github.com/pytorch/examples/blob/master/imagenet/main.py
            base_model = torchvision_models[torchvision_model_type](
                num_classes=self._num_outputs)
            if pretrained:
                pretrained_model = torchvision_models[torchvision_model_type](
                    pretrained=True)
                pretrained_state = pretrained_model.state_dict()
                # Pretrained models typically use Batch Normalization. Since we do
                # not want to collect channel statistics in private learning, we
                # freeze the trained statistics in all batch norm modules, i.e.,
                # the statistics will be from pretrained dataset (ImageNet) instead
                # of private data on device.
                base_model = freeze_batchnorm_modules(base_model)
                base_state = base_model.state_dict()
                state_to_load = {}
                # skip loading the final classifier layer's weight and bias
                for k, v in list(pretrained_state.items())[:-2]:
                    assert k in base_state and v.size() == base_state[k].size()
                    state_to_load[k] = v
                base_model.load_state_dict(state_to_load, strict=False)
                self.base_model = base_model
            else:
                # convert all batch norm module to group norm if not using
                # pretrained models
                self.base_model = convert_batchnorm_modules(base_model)

            # assert there is no batch norm module in current model
            validate_no_batchnorm(self)

        def transform(self, images: torch.Tensor):
            images = (images.float() / 255.0).permute(0, 3, 1, 2)
            if self.training:
                return self.train_transform(images)
            else:
                return self.eval_transform(images)

        def forward(self, images):
            x = self.transform(images)
            return self.base_model(x)

        def loss(self, inputs, targets, eval=False):
            self.eval() if eval else self.train()
            return self.loss_fct(self(inputs), targets)

        @torch.no_grad()
        def metrics(self, inputs, targets, eval=True):
            self.eval() if eval else self.train()
            logits = self(inputs)

            num_data = len(inputs)
            num_predictions = np.ones(self._num_outputs) * num_data
            summed_loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none').sum(dim=0)

            scores = torch.sigmoid(logits)
            predictions = torch.round(scores)
            correct = torch.sum(torch.eq(targets, predictions), dim=0)

            # evaluate precision and recall
            predictions = predictions.bool()
            targets = targets.bool()
            # true positives, positive label predicted as positive
            tps = torch.sum(predictions & targets, dim=0)
            # false positives, negative label predicted as positive
            fps = torch.sum(predictions & ~targets, dim=0)
            # false negatives, positive label predicted as negative
            fns = torch.sum(~predictions & targets, dim=0)
            # micro true positives, false positives, false negatives
            tps_sum = tps.sum().item()
            fps_sum = fps.sum().item()
            fns_sum = fns.sum().item()

            return {
                # Micro metrics: averaged over all predictions
                "micro loss":
                Weighted(summed_loss.sum().item(), num_predictions.sum()),
                "micro accuracy":
                Weighted(correct.sum().item(), num_predictions.sum()),
                "micro precision":
                Weighted(tps_sum, tps_sum + fps_sum),
                "micro recall":
                Weighted(tps_sum, tps_sum + fns_sum),
                "micro AP":
                AveragedPrecision(y_true=targets.cpu().numpy(),
                                  y_pred=scores.cpu().numpy(),
                                  multi_label=False),
                # Macro metrics: averaged over classes
                "macro loss":
                MacroWeighted(summed_loss.cpu().numpy(), num_predictions),
                "macro accuracy":
                MacroWeighted(correct.cpu().numpy(), num_predictions),
                "macro precision":
                MacroWeighted(tps.cpu().numpy(),
                              tps.cpu().numpy() + fps.cpu().numpy()),
                "macro recall":
                MacroWeighted(tps.cpu().numpy(),
                              tps.cpu().numpy() + fns.cpu().numpy()),
                "macro AP":
                AveragedPrecision(y_true=targets.cpu().numpy(),
                                  y_pred=scores.cpu().numpy(),
                                  multi_label=True),
            }

    return MultiLabelCNN(model_type, num_outputs, channel_mean,
                         channel_stddevs, pretrained)


def simple_cnn(input_shape: Tuple[int, ...], num_outputs: int) -> nn.Module:
    """
    A simple CNN with 2 convolutional layers and one dense hidden layer.

    :param input_shape:
        The shape of the input images, e.g. (32,32,3).
    :param num_outputs:
        Size of output softmax layer.
    :return:
        A PyTorch CNN model.
    """
    in_channels = input_shape[-1]
    maxpool_output_size = (input_shape[0] - 4) // 2
    flatten_size = maxpool_output_size * maxpool_output_size * 64

    model = nn.Sequential(*[
        Transpose2D(),
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(flatten_size, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_outputs),
    ])

    # Apply Glorot (Xavier) uniform initialization to match TF2 model.
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    model.loss = types.MethodType(image_classification_loss,
                                  model)  # type: ignore
    model.metrics = types.MethodType(image_classification_metrics,
                                     model)  # type: ignore
    return model
