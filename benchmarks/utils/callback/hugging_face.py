from typing import Tuple

from transformers import PreTrainedModel, PreTrainedTokenizer

from pfl.callback import ModelCheckpointingCallback
from pfl.internal.ops import get_ops
from pfl.metrics import Metrics
from pfl.model.base import StatefulModel


class HuggingFaceModelCheckpointingCallback(ModelCheckpointingCallback):

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
