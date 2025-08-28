# Copyright © 2023-2024 Apple Inc.
import cProfile
import logging
import os
from typing import Optional, Tuple

from pfl.callback.training_process import TrainingProcessCallback
from pfl.metrics import Metrics
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class ProfilerCallback(TrainingProcessCallback):
    """
    Profiles the code using Python's profiler, cProfile.

    A profile is a set of statistics that describes how often and for how long
    various parts of a program are executed.

    This callback can be used to independently profile iterations of an
    algorithm, or to profile all iterations of an algorithm together.

    The profile statistics will be saved as an artifact during training. These
    statistics can be read and analysed using `pstats`:

    .. code-block:: python

        import pstats
        stats = pstats.Stats(<profile-stats-filename>)
        stats.sort_stats(*keys)
        stats.print_stats(*restrictions)

    Alternatively, `SnakeViz` can be used to produce a graphical view of the
    profile in the browser.

    :param frequency:
        Controls frequency and duration of profiling. If `frequency` is an
        integer > 0, profiling is performed per-iteration every `frequency`
        central training iterations. If `frequency` is None, a single
        profile is produced covering all central training iterations.
    :param warmup_iterations:
        Commence profiling after this number of central training iterations.
        If `warmup_iterations` > total number of central iterations, no
        profiling will take place.
    :param dir_name:
        Name of directory in which profiles will be saved. Location
        will be relative to root dir on current platform.
    """

    def __init__(self,
                 frequency: Optional[int] = None,
                 warmup_iterations: int = 0,
                 dir_name: str = 'profile'):
        self._profiler = cProfile.Profile()
        self._cprof_dir = self.platform().create_checkpoint_directories(
            [dir_name])[0]
        self._profiler_frequency = frequency
        self._warmup_iterations = warmup_iterations
        self._profiling_commenced = False

    # For mockability
    def platform(self):
        from pfl.internal.platform.selector import get_platform  # pylint: disable=reimported
        return get_platform()

    def _begin_profiling(self) -> None:
        self._profiler.enable()
        self._profiling_commenced = True

    def _end_profiling(self, filename: str) -> None:
        """
        :param filename
            E.g. 'profile-iteration-0.cprof'.
            Name of file to which profile stats will be saved.
        """
        self._profiler.dump_stats(os.path.join(self._cprof_dir, filename))
        self._profiler = cProfile.Profile()

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if self._warmup_iterations == 0:
            self._begin_profiling()
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Commence profiling of one iteration at end of previous central
        iteration.

        `central_iteration` is zero-indexed.

        `warmup_iterations` is the number of central iterations must have completed
        before profiling begins. This means profiling begins at end of
        iteration when `central_iteration == warmup_iterations - 1`.
        """

        if self._profiler_frequency and self._profiling_commenced:
            if (central_iteration -
                    self._warmup_iterations) % self._profiler_frequency == 0:
                self._end_profiling(
                    f'profile-iteration-{central_iteration}.cprof')

            if (central_iteration - 1 -
                    self._warmup_iterations) % self._profiler_frequency == 0:
                self._begin_profiling()

        if central_iteration == (self._warmup_iterations - 1):
            self._begin_profiling()

        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if not self._profiler_frequency and self._profiling_commenced:
            self._end_profiling('profile.cprof')
