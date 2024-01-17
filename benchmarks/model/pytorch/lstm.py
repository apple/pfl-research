# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import torch
import torch.nn as nn
import torch.nn.functional as F

from pfl.metrics import Weighted
from model.numpy.metrics import Perplexity


class LMBase(nn.Module):

    def __init__(self, pad_symbol: int, unk_symbol: int):
        super().__init__()
        self._pad_symbol = pad_symbol
        self._unk_symbol = unk_symbol

    def loss(self, inputs, targets, eval=False):
        self.eval() if eval else self.train()
        inputs, targets = inputs.long(), targets.long()
        logits = self.forward(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        loss_fct = torch.nn.NLLLoss(reduction='sum',
                                    ignore_index=self._pad_symbol)
        loss = loss_fct(log_probs.reshape(-1, log_probs.shape[-1]),
                        targets.reshape(-1))
        num_tokens = torch.sum(targets != self._pad_symbol)
        return loss / num_tokens

    @torch.no_grad()
    def metrics(self, inputs, targets, eval=False):
        self.eval() if eval else self.train()
        inputs, targets = inputs.long(), targets.long()
        logits = self.forward(inputs)
        log_probs = F.log_softmax(logits, dim=-1)
        loss_fct = torch.nn.NLLLoss(reduction='none',
                                    ignore_index=self._pad_symbol)

        targets = targets.reshape(-1)
        loss = loss_fct(log_probs.reshape(-1, log_probs.shape[-1]), targets)
        correct = logits.argmax(-1).reshape(-1) == targets

        # masks for different symbols
        mask = torch.ones_like(targets, device=inputs.device, dtype=torch.bool)
        pad_mask = mask & (targets != self._pad_symbol)
        unk_mask = pad_mask & (targets != self._unk_symbol)

        # losses ignoring pad and unk symbols
        loss_wo_pad = torch.sum(loss * pad_mask).item()
        loss_wo_unk = torch.sum(loss * unk_mask).item()

        # accuracies ignoring pad and unk symbols
        correct_wo_pad = (correct & pad_mask).float().sum().item()
        correct_wo_unk = (correct & unk_mask).float().sum().item()

        # word count ignoring pad and unk symbols
        num_tokens_wo_pad = pad_mask.float().sum().item()
        num_tokens_wo_unk = unk_mask.float().sum().item()

        return {
            # metrics wo pad symbols
            "loss": Weighted(loss_wo_pad, num_tokens_wo_pad),
            "perplexity": Perplexity(loss_wo_pad, num_tokens_wo_pad),
            "accuracy": Weighted(correct_wo_pad, num_tokens_wo_pad),
            # metrics wo pad and unk symbols
            "loss wo unk": Weighted(loss_wo_unk, num_tokens_wo_unk),
            "perplexity wo unk": Perplexity(loss_wo_unk, num_tokens_wo_unk),
            "accuracy wo unk": Weighted(correct_wo_unk, num_tokens_wo_unk),
        }


class LMLSTM(LMBase):

    def __init__(self, embedding_size: int, num_cell_states: int,
                 num_lstm_layers: int, vocab_size: int, pad_symbol: int,
                 unk_symbol: int):
        super().__init__(pad_symbol, unk_symbol)
        self._embeddings = nn.Embedding(vocab_size,
                                        embedding_size,
                                        padding_idx=pad_symbol)
        self._lstms = nn.ModuleList()

        input_size = embedding_size
        for _ in range(num_lstm_layers):
            self._lstms.append(
                nn.LSTM(input_size, num_cell_states, batch_first=True))
            input_size = num_cell_states

        self._intermediate: torch.nn.Module
        if input_size != embedding_size:
            self._intermediate = nn.Linear(input_size, embedding_size)
        else:
            self._intermediate = nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for name, weight in self._lstms.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(weight)
            if "weight_hh" in name:
                nn.init.orthogonal_(weight)

    def forward(self, inputs):
        x = self._embeddings(inputs)
        for lstm in self._lstms:
            x, _ = lstm(x)
        x = self._intermediate(x)
        logits = x @ self._embeddings.weight.t()
        return logits


def lm_lstm(embedding_size: int, num_cell_states: int, num_lstm_layers: int,
            vocab_size: int, pad_symbol: int, unk_symbol: int):
    """
    Defines a stacked LSTM model with shared input and output embeddings.
    This model is from
    https://github.com/google-research/federated/blob/master/utils/models/stackoverflow_models.py, # pylint: disable=line-too-long
    as used in McMahan et al. (2018) and Konecny et al. (2016). See readme
    for the papers. Confirmed to work with TensorFlow >=2.2.0,<3.

    :param embedding_size:
        Number of parameters in each word embedding vector.
    :param num_cell_states:
        Size of the hidden state of the LSTM
    :param num_lstm_layers:
        Number of LSTM layers stacked on eachother.
    :param vocab_size:
        The size of the input one-hot encodings.
    :param pad_symbol:
        Vocabulary index for PAD symbol.
    :param unk_symbol:
        Vocabulary index for UNK symbol.
    :return:
        A PyTorch LSTM model.
    """
    return LMLSTM(embedding_size, num_cell_states, num_lstm_layers, vocab_size,
                  pad_symbol, unk_symbol)
