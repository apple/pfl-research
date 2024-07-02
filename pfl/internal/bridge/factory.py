# Copyright © 2023-2024 Apple Inc.
from pfl.internal.bridge.base import (
    CommonFrameworkBridge,
    FedProxFrameworkBridge,
    FTRLFrameworkBridge,
    SCAFFOLDFrameworkBridge,
    SGDFrameworkBridge,
)
from pfl.internal.ops.framework_types import MLFramework
from pfl.internal.ops.selector import get_framework_module


class FrameworkBridgeFactory:
    """
    A collection of bridges to deep learning specific
    implementations for several algorithms.
    The bridge returned depends on the Deep Learning
    framework in use.
    This way, we can inject framework-specific code
    into an algorithm, and only have one implementation
    of each algorithm in the public interface, e.g. one
    public FedAvg class instead of one for each of TF,
    PyTorch, etc.

    Each method returns a class with utility functions
    for a particular algorithm.
    """

    @staticmethod
    def common_bridge() -> CommonFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import common as common_pt
            return common_pt.PyTorchCommonBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import common as common_tf
            return common_tf.TFCommonBridge
        elif framework == MLFramework.NUMPY:
            from .numpy import common as common_np
            return common_np.NumpyCommonBridge
        else:
            raise NotImplementedError("Common bridge not available "
                                      f"for framework {framework}")

    @staticmethod
    def sgd_bridge() -> SGDFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import sgd as sgd_pt
            return sgd_pt.PyTorchSGDBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import sgd as sgd_tf
            return sgd_tf.TFSGDBridge
        elif framework == MLFramework.MLX:
            from .mlx import sgd as sgd_mlx
            return sgd_mlx.MLXSGDBridge
        else:
            raise NotImplementedError("SGD bridge not available "
                                      f"for framework {framework}")

    @staticmethod
    def fedprox_bridge() -> FedProxFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import proximal as proximal_pt
            return proximal_pt.PyTorchFedProxBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import proximal as proximal_tf
            return proximal_tf.TFFedProxBridge
        else:
            raise NotImplementedError("FedProx bridge not available "
                                      f"for framework {framework}")

    @staticmethod
    def scaffold_bridge() -> SCAFFOLDFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import scaffold as scaffold_pt
            return scaffold_pt.PyTorchSCAFFOLDBridge
        else:
            raise NotImplementedError("SCAFFOLD bridge not available "
                                      f"for framework {framework}")

    @staticmethod
    def ftrl_bridge() -> FTRLFrameworkBridge:
        framework = get_framework_module().FRAMEWORK_TYPE
        if framework == MLFramework.PYTORCH:
            from .pytorch import ftrl as ftrl_pt
            return ftrl_pt.PyTorchFTRLBridge
        elif framework == MLFramework.TENSORFLOW:
            from .tensorflow import ftrl as ftrl_tf
            return ftrl_tf.TFFTRLBridge
        else:
            raise NotImplementedError("FTRL bridge not available "
                                      f"for framework {framework}")
