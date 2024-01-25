# Copyright © 2023-2024 Apple Inc.

from pfl.internal.ops.selector import (
    _internal_reset_framework_module,
    get_framework_module,
    has_framework_module,
    set_framework_module,
)


def test_ops_meta():
    _internal_reset_framework_module()

    class TestModule:
        pass

    module = TestModule()

    set_framework_module(module)
    assert has_framework_module()
    assert get_framework_module() == module
    _internal_reset_framework_module()
