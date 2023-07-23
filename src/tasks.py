import torch.nn as nn
from torch import Tensor

from src.model import GPT


class GPTPretrain(nn.Module):
    def __init__(
        self,
        gpt_model: GPT,
        vocab_size: int,
        d_hidden: int,
    ) -> None:
        super().__init__()
        self.gpt = gpt_model

        self.transpose_embedding = nn.Linear(d_hidden, vocab_size, bias=False)
        self.transpose_embedding.weight = self.gpt.decoder.src_emb.weight

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_outputs = self.gpt(dec_inputs)  # -> [bs, max_seq_size, d_hidden]
        logit_lm = self.transpose_embedding(dec_outputs)  # -> [bs, max_seq_size, vocab]
        return logit_lm  # -> [bs, max_seq_size, vocab_size]


class NaverClassification(nn.Module):
    def __init__(self, gpt_model: GPT, d_hidden: int, n_outputs: int) -> None:
        super().__init__()
        self.gpt = gpt_model
        self.project_cls = nn.Linear(d_hidden, n_outputs, bias=False)

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_outputs = self.gpt(dec_inputs)  # -> [bs, max_seq_size, d_hidden]
        dec_outputs = dec_outputs[:, -1].contiguous()  # -> [bs, d_hidden], 마지막 토큰의 output을 사용해서 분류 
        return self.project_cls(dec_outputs)  # -> [bs, n_outputs]
