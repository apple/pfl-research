# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from pfl.internal.ops.selector import (
    _internal_reset_framework_module,
    get_default_framework_module,
    get_framework_module,
    has_framework_module,
    set_framework_module,
)


def test_no_framework_module():
    _internal_reset_framework_module()
    assert not has_framework_module()
    ops = get_default_framework_module()
    assert 'add_gaussian_noise' in ops.__dict__
    assert not has_framework_module()


def test_set_framework_module():
    _internal_reset_framework_module()

    module = ['dummy']
    set_framework_module(module)

    assert has_framework_module()
    assert get_framework_module() is module
    assert get_default_framework_module() is module
    _internal_reset_framework_module()
