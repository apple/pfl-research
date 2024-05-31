# -*- coding: utf-8 -*-

from typing import Dict

import torch
import torch.nn as nn

from pfl.metrics import Weighted


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                  reduction: str) -> torch.Tensor:
    """ PyTorch cross entropy loss """
    # TODO: support logits with more than 2 dimensions
    assert logits.ndim == 2, f"expect 2D tensor, get {logits.ndim}D"
    loss_fct = nn.CrossEntropyLoss(reduction=reduction)
    if targets.dim() > 1:
        targets = targets.squeeze()
    return loss_fct(logits, targets.long())


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ PyTorch classification accuracy """
    # TODO: support logits with more than 2 dimensions
    assert logits.ndim == 2, f"expect 2D tensor, get {logits.ndim}D"
    correct = logits.argmax(-1) == targets.squeeze().long()
    return correct.float().sum()


def image_classification_loss(self: nn.Module,
                              inputs: torch.Tensor,
                              targets: torch.Tensor,
                              eval: bool = False) -> torch.Tensor:
    """ Loss function to be attached to `PyTorchModel` for classification """
    self.eval() if eval else self.train()
    return cross_entropy(self(inputs), targets, "mean")


@torch.no_grad()
def image_classification_metrics(self: nn.Module,
                                 inputs: torch.Tensor,
                                 targets: torch.Tensor,
                                 eval: bool = True) -> Dict[str, Weighted]:
    """ Metrics function to be attached to `PyTorchModel` for classification """
    self.eval() if eval else self.train()
    logits = self(inputs)
    num_samples = len(inputs)
    correct = accuracy(logits, targets).item()
    loss = cross_entropy(logits, targets, "sum").item()
    return {
        "loss": Weighted(loss, num_samples),
        "accuracy": Weighted(correct, num_samples)
    }
