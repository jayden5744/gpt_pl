import torch.nn as nn
from torch import Tensor

from src.model import GPT


class GPTPretrain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        max_sequence_size: int,
        padding_id: int,
        dropout_rate: float
        ) -> None:
        super().__init__()
        self.gpt = GPT(
            vocab_size=vocab_size,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            max_sequence_size=max_sequence_size,
            padding_id=padding_id,
            dropout_rate=dropout_rate
        )

        self.transpose_embedding = nn.Linear(d_hidden, vocab_size, bias=False)
        self.transpose_embedding.weight = self.gpt.decoder.src_emb.weight

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_outputs = self.gpt(dec_inputs)  # -> [bs, max_seq_size, d_hidden]
        logit_lm = self.transpose_embedding(dec_outputs)  # -> [bs, max_seq_size, vocab]
        return logit_lm # -> [bs, max_seq_size, vocab_size]