# Copyright © 2023-2024 Apple Inc.
from typing import Callable, Type

from torch import nn

from .layer import FrozenBatchNorm1D, FrozenBatchNorm2D, FrozenBatchNorm3D


def _replace_child(root: nn.Module, child_name: str,
                   converter: Callable[[nn.Module], nn.Module]) -> None:
    """
    Converts a sub-module to a new module given a helper
    function, the root module and a string representing
    the name of the submodule to be replaced.
    """
    # find the immediate parent
    parent = root
    nameList = child_name.split(".")
    for name in nameList[:-1]:
        new_parent = parent._modules[name]
        assert new_parent is not None
        parent = new_parent
    # set to identity
    assert parent is not None
    parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])


def replace_all_modules(
    root: nn.Module,
    target_class: Type[nn.Module],
    converter: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Converts all the submodules (of root) that have the same
    type as target_class, given a converter, a module root,
    and a target class type.
    """
    # base case
    if isinstance(root, target_class):
        return converter(root)

    for name, obj in root.named_modules():
        if isinstance(obj, target_class):
            _replace_child(root, name, converter)
    return root


def _batchnorm_to_groupnorm(
        module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm ``module`` to GroupNorm module.
    This is a helper function.

    Notes:
        A default value of 32 is chosen for the number of groups based on the
        paper *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
        https://arxiv.org/pdf/1706.02677.pdf
    """
    return nn.GroupNorm(min(32, module.num_features),
                        module.num_features,
                        affine=True)


def _batchnorm_to_freeze_batchnorm(
        module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm module to the corresponding FrozenBatchNorm module.
    This is useful for private finetuning models with BatchNorm module since
    we do not want to collect training data statistics for updating BatchNorm
    parameters. Instead, the statistics of FrozenBatchNorm is never updated.
    """

    def match_dim():
        if isinstance(module, nn.BatchNorm1d):
            return FrozenBatchNorm1D
        elif isinstance(module, nn.BatchNorm2d):
            return FrozenBatchNorm2D
        elif isinstance(module, nn.BatchNorm3d):
            return FrozenBatchNorm3D

    return match_dim()(num_features=module.num_features,
                       eps=module.eps,
                       momentum=module.momentum,
                       affine=module.affine,
                       track_running_stats=module.track_running_stats)


def validate_no_batchnorm(module: nn.Module):
    """
    Assert no regular batch normalization in model architecture.
    """
    ans = not isinstance(module, nn.modules.batchnorm._BatchNorm)
    for child in module.children():
        ans = ans and validate_no_batchnorm(child)
    assert ans
    return ans


def convert_batchnorm_modules(
    model: nn.Module,
    converter: Callable[[nn.modules.batchnorm._BatchNorm],
                        nn.Module] = _batchnorm_to_groupnorm,
) -> nn.Module:
    """
    Converts all BatchNorm modules to another module
    (defaults to GroupNorm) that is privacy compliant.

    :param model
        Module instance, potentially with sub-modules
    :param converter
        Function or a lambda that converts an instance of a
        Batchnorm to another nn.Module.

    :return
        Model with all the BatchNorm types replaced by another operation
        by using the provided converter, defaulting to GroupNorm if one
        isn't provided.
    """
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm,
                               converter)


def freeze_batchnorm_modules(model: nn.Module):
    """
    Convert all BatchNorm modules to FrozenBatchNorm modules where the stats
    are frozen and not updated during training.

    :param model
        Module instance, potentially with sub-modules
    :return
        A `FrozenBatchNorm` module with the same dimension as input module.
    """
    return convert_batchnorm_modules(model, _batchnorm_to_freeze_batchnorm)
