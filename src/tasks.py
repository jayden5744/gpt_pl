
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

        self.projection_lm = nn.Linear(d_hidden, vocab_size, bias=False)
        self.projection_lm.weight = self.gpt.decoder.src_emb.weight

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_ouputs = self.gpt(dec_inputs)

        logits_lm = self.projection_lm(dec_ouputs) # -> [bs, max_seq_size, vocab]
        return logits_lm[:, :-1, :].contiguous() # -> [bs, max_seq_size -1, vocab_size]



class GPTClassification(GPTPretrain):
    def __init__(self, vocab_size: int, d_hidden: int, n_layers: int, n_heads: int, ff_dim: int, max_sequence_size: int, padding_id: int, dropout_rate: float, n_output: int) -> None:
        super().__init__(vocab_size, d_hidden, n_layers, n_heads, ff_dim, max_sequence_size, padding_id, dropout_rate)
        self.projection_cls = nn.Linear(self.d_hidden, n_output, bias=False)

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_ouputs = self.gpt(dec_inputs)                       # -> [bs, max_seq_size, d_hidden]
        logits_lm = self.projection_lm(dec_ouputs)              # -> [bs, max_seq_size, vocab]
        
        dec_outputs = dec_outputs[:, -1].contiguous()           # -> [bs, d_hidden]
        logits_cls = self.projection_cls(dec_outputs)           # -> [bs, n_output]
        
        return logits_lm[:, :-1, :].contiguous(), logits_cls  # (bs, n_dec_seq - 1, n_dec_vocab), (bs, n_output)
