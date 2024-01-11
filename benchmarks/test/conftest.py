# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import pytest


def pytest_addoption(parser):
    parser.addoption('--macos',
                     action='store_true',
                     default=False,
                     help='run tests that require MacOS')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--macos'):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason='need --macos option to run')
    for item in items:
        if 'macos' in item.keywords:
            item.add_marker(skip_slow)
