# Copyright © 2023-2024 Apple Inc.
"""
Select a framework (e.g. TensorFlow) with operations to use.
The framework can be set only once during a run of a program.
"""
from . import numpy_ops

_framework_module = None


def _internal_reset_framework_module():
    global _framework_module
    _framework_module = None


def get_framework_module():
    global _framework_module  # pylint: disable=global-variable-not-assigned
    assert _framework_module is not None
    return _framework_module


def get_default_framework_module():
    global _framework_module  # pylint: disable=global-variable-not-assigned
    if _framework_module is None:
        return numpy_ops
    else:
        return _framework_module


def set_framework_module(module, old_module=None):
    """
    Set a framework module.
    Set `old_module` if you are certain that you need to change the module from
    another one.

    :Example:

    .. code-block:: python

        from pfl.internal.ops import pytorch_ops
        set_framework_module(pytorch_ops)
    """
    global _framework_module
    assert (_framework_module is None or _framework_module is old_module
            or _framework_module is module), (
                "Attempt to use multiple frameworks within one process")
    _framework_module = module


def has_framework_module():
    """
    Return `True` iff a framework has been set.
    If not, also make sure that it cannot be set at a later time.
    ``numpy_ops`` does not count as a framework so False will be returned
    if the framework is set to ``numpy_ops``.
    """
    global _framework_module
    if _framework_module is numpy_ops:
        return False
    if _framework_module is None:
        _framework_module = numpy_ops
        return False
    return True
