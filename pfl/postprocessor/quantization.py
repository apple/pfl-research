# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Optional, Tuple

import numpy as np

from pfl.context import UserContext
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.postprocessor.base import Postprocessor
from pfl.stats import StatisticsType

logger = logging.getLogger(name=__name__)


class QuantizationPostprocessor(Postprocessor):
    """
    Quantizes client statistics to low-bit representations (e.g. 4-bit or 8-bit)
    before communication back to the central server, reducing edge-to-server
    bandwidth consumption in federated learning.

    Supports block-wise dynamic scaling and optional unbiased stochastic rounding.

    :param nbits:
        Number of quantization bits (between 2 and 8). Defaults to 4.
    :param block_size:
        Number of elements per quantization block. If None, scales the entire
        tensor with a single dynamic scale factor. Defaults to 64.
    :param stochastic:
        Whether to apply stochastic rounding (unbiased gradient quantization)
        or deterministic nearest-integer rounding. Defaults to False.
    :param min_elements:
        Minimum number of elements in a tensor to qualify for quantization.
        Tensors with fewer elements remain untouched. Defaults to 64.
    """

    def __init__(self,
                 nbits: int = 4,
                 block_size: Optional[int] = 64,
                 stochastic: bool = False,
                 min_elements: int = 64):
        if not isinstance(nbits, int) or isinstance(nbits, bool) or nbits < 2 or nbits > 8:
            raise ValueError(f"nbits must be an integer between 2 and 8, got {nbits}")
        if block_size is not None:
            if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
                raise ValueError(f"block_size must be a positive integer or None, got {block_size}")
        if not isinstance(min_elements, int) or isinstance(min_elements, bool) or min_elements < 0:
            raise ValueError(f"min_elements must be a non-negative integer, got {min_elements}")

        self._nbits = nbits
        self._block_size = block_size
        self._stochastic = stochastic
        self._min_elements = min_elements
        self._max_q = (1 << (nbits - 1)) - 1
        self._min_q = -(1 << (nbits - 1))

    @property
    def nbits(self) -> int:
        return self._nbits

    @property
    def block_size(self) -> Optional[int]:
        return self._block_size

    @property
    def stochastic(self) -> bool:
        return self._stochastic

    @property
    def min_elements(self) -> int:
        return self._min_elements

    def _quantize_array(self, arr: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        if not np.issubdtype(arr.dtype, np.floating) or arr.size < self._min_elements:
            return arr

        if not np.all(np.isfinite(arr)):
            raise ValueError("Input array contains non-finite values (NaN or Inf).")

        orig_shape = arr.shape
        flat = arr.flatten().astype(np.float32)
        n = flat.size

        if self._block_size is None or self._block_size >= n:
            max_abs = float(np.max(np.abs(flat)))
            if max_abs == 0.0:
                return np.zeros_like(arr)
            scale = max_abs / float(self._max_q)
            normalized = flat / scale
            if self._stochastic:
                rng = np.random.default_rng(seed)
                normalized += rng.uniform(-0.5, 0.5, size=normalized.shape)
            q = np.clip(np.round(normalized), self._min_q, self._max_q)
            restored = (q * scale).astype(arr.dtype)
            return restored.reshape(orig_shape)

        block_size = self._block_size
        pad_len = (block_size - (n % block_size)) % block_size
        padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat

        blocks = padded.reshape(-1, block_size)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        zero_mask = max_abs == 0.0
        scale = max_abs / float(self._max_q)
        safe_scale = np.where(zero_mask, 1.0, scale)

        normalized = blocks / safe_scale
        if self._stochastic:
            rng = np.random.default_rng(seed)
            normalized += rng.uniform(-0.5, 0.5, size=normalized.shape)

        q = np.where(zero_mask, 0.0, np.clip(np.round(normalized), self._min_q, self._max_q))
        restored_blocks = q * scale
        rec_flat = restored_blocks.flatten()
        if pad_len > 0:
            rec_flat = rec_flat[:-pad_len]
        return rec_flat.reshape(orig_shape).astype(arr.dtype)

    def postprocess_one_user(
            self, *, stats: StatisticsType,
            user_context: UserContext) -> Tuple[StatisticsType, Metrics]:
        """
        Quantizes client training statistics before communication back to server.
        """
        seed = user_context.seed
        total_sq_error = 0.0
        total_elements = 0

        def _transform(weights):
            nonlocal total_sq_error, total_elements
            quantized_weights = []
            for w in weights:
                qw = self._quantize_array(w, seed=seed)
                if w is not qw and np.issubdtype(w.dtype, np.floating) and w.size >= self._min_elements:
                    diff = w.astype(np.float64) - qw.astype(np.float64)
                    total_sq_error += float(np.sum(diff * diff))
                    total_elements += w.size
                quantized_weights.append(qw)
            return quantized_weights

        quantized_stats = stats.apply(_transform)
        metrics = Metrics()
        metrics[StringMetricName("quantization | nbits")] = self._nbits
        if total_elements > 0:
            mse = total_sq_error / total_elements
            metrics[StringMetricName("quantization | mse")] = Weighted.from_unweighted(
                mse, weight=total_elements)

        return quantized_stats, metrics
