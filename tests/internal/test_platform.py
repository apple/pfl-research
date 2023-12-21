# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from pfl.internal.platform import generic_platform


class TestGenericPlatform:

    def test_get_distributed_addresses(self):
        platform = generic_platform.GenericPlatform()
        data = platform.get_distributed_addresses()

        assert (data[0] == 0)
        assert (data[1] is None)
