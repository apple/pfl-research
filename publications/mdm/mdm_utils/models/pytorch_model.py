import types
from typing import Tuple, Dict

import torch
from torch import nn

from pfl.metrics import Weighted

# Taken directly from the Ramsay Examples Repo, all methods needed to define the
# pytorch model used for training.


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

    model.loss = types.MethodType(image_classification_loss, model)
    model.metrics = types.MethodType(image_classification_metrics, model)
    return model


class Transpose2D(nn.Module):
    """
    Transpose Tensorflow style image to PyTorch compatible
    """

    def forward(self, inputs: torch.Tensor):
        return inputs.permute((0, 3, 1, 2))


def image_classification_loss(self: nn.Module,
                              inputs: torch.Tensor,
                              targets: torch.Tensor,
                              eval: bool = False) -> torch.Tensor:
    """ Loss function to be attached to `PyTorchModel` for classification """
    self.eval() if eval else self.train()
    return cross_entropy(self(inputs), targets, "mean")


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
