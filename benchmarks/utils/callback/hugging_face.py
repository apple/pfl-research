# Copyright © 2023-2024 Apple Inc.

from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from pfl.callback import ModelCheckpointingCallback
from pfl.internal.ops import get_ops
from pfl.metrics import Metrics
from pfl.model.base import StatefulModel


class HuggingFaceModelCheckpointingCallback(ModelCheckpointingCallback):
    """
    Callback to save Hugging Face model checkpoints.

    :param model:
        A Hugging Face `transformers.PreTrainedModel` to be saved.
    :param tokenizer:
        A Hugging Face `transformers.PreTrainedTokenizer` to be saved.
    :param model_checkpoint_dir:
        A path to disk for saving the trained model. Location
        will be relative to root dir on current platform.
    :param checkpoint_frequency:
        The number of central iterations after which to save a model.
        When zero (the default), the model is saved once after
        training is complete.
    """

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 model_checkpoint_dir: str,
                 checkpoint_frequency: int = 0):
        super().__init__(model_checkpoint_dir,
                         checkpoint_frequency=checkpoint_frequency)
        self._model = model
        self._tokenizer = tokenizer

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: StatefulModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if get_ops(
        ).distributed.local_rank == 0 and self._should_checkpoint_now(
                central_iteration):
            self._model.save_pretrained(self.model_checkpoint_dir)
            self._tokenizer.save_pretrained(self.model_checkpoint_dir)
        return False, Metrics()

    def on_train_end(self, *, model: StatefulModel) -> None:
        if get_ops(
        ).distributed.local_rank == 0 and self.checkpoint_frequency == 0:
            self._model.save_pretrained(self.model_checkpoint_dir)
            self._tokenizer.save_pretrained(self.model_checkpoint_dir)
