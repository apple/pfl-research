# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.


class PFLError(Exception):
    """ Base error class """
    pass


class UserNotFoundError(PFLError):

    def __init__(self, payload):
        PFLError.__init__(self, f"User {payload} not found.")


class CheckpointNotFoundError(PFLError):

    def __init__(self, path):
        PFLError.__init__(self, f"Checkpoint not found at location {path}.")


class MatrixFactorizationError(PFLError):
    """
    Exception for linear algebra error in matrix factorization.
    """
