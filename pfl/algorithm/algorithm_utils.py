# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import asyncio
from typing import List, Tuple

from pfl.aggregate.base import Backend
from pfl.context import CentralContext
from pfl.metrics import Metrics
from pfl.model.base import Model
from pfl.stats import StatisticsType


def run_train_eval(
    algorithm,
    backend: Backend,
    model: Model,
    central_contexts: Tuple[CentralContext, ...],
) -> List[Tuple[StatisticsType, Metrics]]:
    """
    Run training/evaluation and gather aggregated model updates and metrics
    for multiple rounds given multiple contexts.
    Used in ``FederatedAlgorithm.run``.

    :param algorithm:
        The :class:`~pfl.algorithm.base.FederatedAlgorithm` to use.
        Implements local training behaviour and central model update behaviour.
        Note that some algorithms are incompatible with some models.
    :param backend:
        The :class:`~pfl.aggregate.base.Backend` to use to
        distribute training of individual users and aggregate the results.
        Can be used for simulation or live training with the infrastructure.
    :param model:
        The model to train.
    :param central_contexts:
        A tuple of multiple contexts. Gather aggregated results once for each
        of the contexts.
    :returns:
        Aggregated model updates and metrics.
    """

    async def run_iteration(central_context, initial_sleep: bool):
        """
        Perform gathering results for a given central context.
        """
        if initial_sleep:
            # This is needed to ensure that the iteration with no sleep
            # is ordered to start before other iterations with initial sleep.
            # Which is needed for synchronous worker communication in
            # distributed simulations.
            await asyncio.sleep(0)

        return await backend.async_gather_results(
            model=model,
            training_algorithm=algorithm,
            central_context=central_context)

    async def run_all_iterations():
        return await asyncio.gather(*[
            run_iteration(central_context, initial_sleep=index == 0)
            for index, central_context in enumerate(central_contexts)
        ])

    return asyncio.get_event_loop().run_until_complete(run_all_iterations())
