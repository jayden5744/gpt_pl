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
        return logit_lm[:, :-1, :].contiguous()  # -> [bs, max_seq_size-1, vocab_size]


class GPTClassification(nn.Module):
    def __init__(self, gpt_model: GPT, d_hidden: int, n_outputs: int) -> None:
        super().__init__()
        self.gpt = gpt_model
        self.project_cls = nn.Linear(d_hidden, n_outputs, bias=False)

    def forward(self, dec_inputs: Tensor) -> Tensor:
        dec_outputs = self.gpt(dec_inputs)  # -> [bs, max_seq_size, d_hidden]
        dec_outputs = dec_outputs[:, -1].contiguous()  # -> [bs, d_hidden], 마지막 토큰의 output을 사용해서 분류 
        return self.project_cls(dec_outputs)  # -> [bs, n_outputs]

    def generate(self, dec_inputs: Tensor):
        model.eval()
        dec_outputs = self.gpt(dec_inputs)  # -> [bs, max_seq_size, d_hidden]
        dec_outputs = dec_outputs[:, -1].contiguous()
        logits = self.project_cls(dec_outputs)
        logit_probs = nn.functional.softmax(logits, dim=-1)
        print(logit_probs)




class GPTSimilarity(nn.Module):
    def __init__(self, gpt_model: GPT, d_hidden: int, n_outputs: int = 2):
        super().__init__()
        self.gpt = gpt_model
        self.project_cls = nn.Linear(d_hidden, n_outputs, bias=False)

    def forward(self, first_sentence:Tensor, second_sentence: Tensor) -> Tensor:
        first_tensor = self.gpt(first_sentence)
        second_tensor = self.gpt(second_sentence)
        dec_output = first_tensor + second_sentence
        dec_outputs = dec_outputs[:, -1].contiguous()  # -> [bs, d_hidden]
        return self.project_cls(dec_outputs)  # -> [bs, n_outputs]

