# Copyright © 2023-2024 Apple Inc.
import logging
import os
from collections import OrderedDict
from typing import Dict, Tuple

from pfl.callback.training_process import TrainingProcessCallback
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class AggregateMetricsToDisk(TrainingProcessCallback):
    """
    Callback to write aggregated metrics to disk with a given frequency
    with respect to the number of central iterations.

    :param output_path:
        Path to where the csv file of aggregated metrics should be written
        relative to the root dir on current platform.
    :param frequency:
        Write aggregated metrics to file every ``frequency`` central
        iterations.
        Can be useful to skip iterations where no evaluation is done if that
        is also set at a frequency.
    :param check_existing_file:
        Throw error if ``output_path`` already exists and you don't want to
        overwrite it.
    """

    def __init__(self,
                 output_path: str,
                 frequency: int = 1,
                 decimal_points: int = 6,
                 check_existing_file: bool = False):
        dir_name = os.path.dirname(output_path)
        file_name = os.path.basename(output_path)
        platform_dir = self.platform().create_checkpoint_directories(
            [dir_name])[0]
        output_path = os.path.join(platform_dir, file_name)
        self._frequency = frequency
        self._decimal_points = decimal_points
        self._output_path = output_path
        self._columns_to_index: Dict = OrderedDict()

        if get_ops().distributed.local_rank == 0:
            if check_existing_file:
                assert not os.path.exists(
                    output_path), "File {output_path} already exists"
            self._fp = open(self._output_path, 'w')  # noqa: SIM115

    # For mockability
    def platform(self):
        from pfl.internal.platform.selector import get_platform
        return get_platform()

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if get_ops().distributed.local_rank != 0:
            return Metrics()
        self._columns_to_index = OrderedDict()
        self._columns_to_index['central_iteration'] = 0
        # Empty header, update later.
        self._fp.write('central_iteration,\n')
        return Metrics()

    def _update_header(self):
        """ Replace header with current known columns """
        self._fp.flush()
        self._fp.close()
        with open(self._output_path) as f:
            lines = f.readlines()
        num_columns_before = len(lines[0].split(','))
        lines[0] = ','.join(self._columns_to_index.keys()) + '\n'
        # Append empty values on all rows for new columns.
        for i in range(1, len(lines)):
            lines[i] = (
                lines[i].strip() + ',' *
                (len(self._columns_to_index.keys()) - num_columns_before) +
                '\n')

        with open(self._output_path, 'w') as f:
            f.writelines(lines)
        self._fp = open(self._output_path, 'a')  # noqa: SIM115

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if central_iteration % self._frequency != 0:
            # Don't write to disk this central iteration.
            return False, Metrics()
        if get_ops().distributed.local_rank != 0:
            return False, Metrics()

        raw_metrics = aggregate_metrics.to_simple_dict()
        raw_metrics['central_iteration'] = central_iteration
        columns_changed = False
        for name in raw_metrics:
            if name not in self._columns_to_index:
                next_ix = len(self._columns_to_index)
                self._columns_to_index[name] = next_ix
                columns_changed = True
        if columns_changed:
            self._update_header()

        line = ','.join(
            str(round(raw_metrics[c], self._decimal_points)) if c in
            raw_metrics else '' for c in self._columns_to_index)
        self._fp.write(line + '\n')
        self._fp.flush()
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if get_ops().distributed.local_rank == 0:
            self._fp.flush()
            self._fp.close()
