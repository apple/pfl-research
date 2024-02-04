# Copyright © 2023-2024 Apple Inc.
import logging

from pfl.internal.platform import generic_platform

logger = logging.getLogger(name=__name__)
_platform = None


def get_platform():
    """
    Get the current platform. If the platform is not set, return
    the default platform ``pfl.internal.platform.GenericPlatform``.
    """
    global _platform

    if _platform is not None:
        return _platform
    else:
        _platform = generic_platform.GenericPlatform()

    return _platform


def set_platform(module: generic_platform.Platform):
    """
    Manually set the platform module. Can be useful when
    you are running on an unsupported platform and made
    a custom integration for it.
    """
    global _platform
    logger.info(f'Manually changing platform from {_platform} to {module}')
    _platform = module
