# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import numpy as np


def positional_encoding(length: int, depth: int) -> np.ndarray:
    """
    Positional encoding as in Section 3.5 of
    https://arxiv.org/pdf/1706.03762.pdf
    :param length:
        Maximum length of the positional encoding.
    :param depth:
        Dimension of the positional encoding.
    :return:
        A numpy array of shape (length, depth).
    """
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    div_term = np.exp(np.arange(0, depth, 2) * (-np.log(10000.0) / depth))
    pe = np.zeros((length, depth), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe
