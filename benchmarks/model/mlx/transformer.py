# Copyright © 2024 Apple Inc.
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from pfl.metrics import Weighted

from model.numpy.layer import positional_encoding
from model.numpy.metrics import Perplexity


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, max_sequence_length: int):
        super().__init__()
        # Start with underscore so it is not included in the parameters
        self._pe = mx.array(positional_encoding(max_sequence_length, embedding_size))

    def __call__(self, x):
        return x + mx.expand_dims(self._pe[:x.shape[1], :], 0)


class LMTransformer(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, num_heads: int,
                 feedforward_size: int, num_transformer_layers: int,
                 dropout_rate: float, vocab_size: int,
                 max_sequence_length: int, pad_symbol: int, unk_symbol: int):
        super().__init__()
        self._pad_symbol = pad_symbol
        self._unk_symbol = unk_symbol
        self._embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pe = PositionalEncoding(embedding_size, max_sequence_length)
        self.layers = []
        for _ in range(num_transformer_layers):
            l = nn.TransformerEncoderLayer(
                hidden_size, num_heads, feedforward_size, dropout_rate, norm_first=False
            )
            # Need to re-init multi-head attention with bias to match network of PyTorch and TF.
            l.attention = nn.MultiHeadAttention(hidden_size, num_heads, bias=True)
            self.layers.append(l)

        self._proj_in = nn.Linear(embedding_size, hidden_size) if embedding_size != hidden_size else nn.Identity()
        self._proj_out = nn.Linear(hidden_size, embedding_size) if embedding_size != hidden_size else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        embedding_w = mx.random.uniform(-0.05, 0.05, shape=self.embedding.trainable_parameters()['weight'].shape)
        self.embedding.update({'weight':embedding_w})

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.trainable_parameters()))
        return nparams

    def __call__(self, inputs):
        L = inputs.shape[1]
        casual_mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(inputs) * math.sqrt(self._embedding_size)
        x = self.pe(x)
        x = self._proj_in(x)
        for l in self.layers:
            x = l(x, casual_mask)
        x = self._proj_out(x)
        logits = x @ self.embedding.weight.T
        return logits

    def loss(self, inputs, targets, is_eval=False):
        self.eval() if is_eval else self.train()
        logits = self(inputs)
        mask = (targets != self._pad_symbol).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction='none')
        loss = mx.sum(losses * mask)
        num_tokens = mx.sum(mask)
        # Above is equivalent to this but ignoring padding labels for loss.
        #loss = nn.losses.cross_entropy(logits, targets, reduction='sum')
        #num_tokens = mx.sum(targets != self._pad_symbol)
        return loss / num_tokens

    def metrics(self, inputs, targets, eval=True):
        self.eval() if eval else self.train()
        logits = self(inputs)
        loss = nn.losses.cross_entropy(logits, targets, reduction='none').reshape(-1)
        targets = targets.reshape(-1)
        correct = mx.argmax(logits, axis=-1).reshape(-1) == targets

        mask = mx.ones_like(targets).astype(mx.bool_)
        pad_mask = mask & (targets != self._pad_symbol)
        unk_mask = pad_mask & (targets != self._unk_symbol)

        loss_wo_pad = mx.sum(loss * pad_mask).item()
        loss_wo_unk = mx.sum(loss * unk_mask).item()

        correct_wo_pad = (correct & pad_mask).astype(mx.float32).sum().item()
        correct_wo_unk = (correct & unk_mask).astype(mx.float32).sum().item()

        num_tokens_wo_pad = pad_mask.astype(mx.float32).sum().item()
        num_tokens_wo_unk = unk_mask.astype(mx.float32).sum().item()

        return {
            "loss": Weighted(loss_wo_pad, num_tokens_wo_pad),
            "perplexity": Perplexity(loss_wo_pad, num_tokens_wo_pad),
            "accuracy": Weighted(correct_wo_pad, num_tokens_wo_pad),
            "loss wo unk": Weighted(loss_wo_unk, num_tokens_wo_unk),
            "perplexity wo unk": Perplexity(loss_wo_unk, num_tokens_wo_unk),
            "accuracy wo unk": Weighted(correct_wo_unk, num_tokens_wo_unk),
        }


def lm_transformer(embedding_size: int,
                   hidden_size: int,
                   num_heads: int,
                   feedforward_size: int,
                   num_transformer_layers: int,
                   vocab_size: int,
                   max_sequence_length: int,
                   pad_symbol: int,
                   unk_symbol: int,
                   dropout_rate: float = 0.1):
    """
    Defines a Transformer model with shared input and output embeddings.
    This model is from
    https://github.com/google-research/federated/blob/master/fedopt_guide/stackoverflow_transformer/transformer_models.py
    as used in Wang et al. (2021). See readme for the papers.

    :param embedding_size:
        Number of parameters in each word embedding vector.
    :param hidden_size:
        Size of the hidden state of the Transformer.
    :param num_heads:
        Number of heads in the multi-head attention layer.
    :param feedforward_size:
        Size of the feed forward layer in the Transformer.
    :param num_transformer_layers:
        Number of Transformer layers stacked on eachother.
    :param vocab_size:
        The size of the input one-hot encodings.
    :param max_sequence_length:
        Sequence length to decide the size of positional encoding.
    :param pad_symbol:
        Vocabulary index for PAD symbol.
    :param unk_symbol:
        Vocabulary index for UNK symbol.
    :param dropout_rate:
        Dropout rate applied in the Transformer model.
    :return:
        A PyTorch Transformer model.
    """
    return LMTransformer(embedding_size, hidden_size, num_heads,
                         feedforward_size, num_transformer_layers,
                         dropout_rate, vocab_size, max_sequence_length,
                         pad_symbol, unk_symbol)
