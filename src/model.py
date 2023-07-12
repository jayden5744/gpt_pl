from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_position_encoding_table(max_sequence_size: int, d_hidden: int) -> Tensor:
    def get_angle(position: int, i: int) -> float:
        return position / np.power(10000, 2 * (i // 2) / d_hidden)

    def get_angle_vector(position: int) -> List[float]:
        return [get_angle(position, hid_j) for hid_j in range(d_hidden)]

    pe_table = Tensor([get_angle_vector(pos_i) for pos_i in range(max_sequence_size)])
    pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
    pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i +1
    return pe_table


def get_position(inputs: Tensor) -> Tensor:
    position = (
        torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype)
        .expand(inputs.size(0), inputs.size(1))
        .contiguous()
    )  # -> [bs, max_seq_size]
    return position


def get_padding_mask(inputs: Tensor, padding_id: int) -> Tensor:
    """padding token에 mask를 씌우는 함수

    Args:
        input_tensor (Tensor): 입력문장, [batch_size, seq_len]
        padding_id (int): padding id

    Returns:
        Tensor: 입력문장 padding 포함여부 [batch_size, seq_len, seq_len]
    """
    pad_attn_mask = inputs.data.eq(padding_id).unsqueeze(
        1
    )  # => [batch_size, 1, len_k]  True / False
    return pad_attn_mask.expand(
        inputs.size(0), inputs.size(1), inputs.size(1)
    ).contiguous()  # => [batch_size, len_q, len_k]


def get_look_ahead_mask(inputs: Tensor) -> Tensor:
    """look ahead mask 생성 함수
    자기 자신보다 미래에 있는 단어들을 참고할 수 없도록 마스킹하는 함수
    Args:
        dec_input (Tensor): 입력문장, [batch_size, seq_len]
    Returns:
        Tensor: _description_
    """
    look_ahead_mask = (
        torch.ones_like(inputs)
        .unsqueeze(-1)
        .expand(inputs.size(0), inputs.size(1), inputs.size(1))
    )  # => [batch_size, seq_len, seq_len]
    look_ahead_mask = look_ahead_mask.triu(
        diagonal=1
    )  # upper triangular part of a matrix(2-D) => [batch_size, seq_len, seq_len]
    return look_ahead_mask.eq(1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = head_dim**0.5

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attion_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            query (Tensor): (bs, n_heads, max_seq, d_hidden)
            key (Tensor): (bs, n_heads, max_seq, d_hidden)
            value (Tensor): (bs, n_heads, max_seq, d_hidden)
            attion_mask (Tensor): (bs, n_heads, max_seq, max_seq)

        Returns:
            Tuple[Tensor, Tensor]: _description_
        """
        scores = (
            torch.matmul(query, key.transpose(-1, -2)) / self.scale
        )  # => [bs, n_heads, len_q(=max_seq), len_k(=max_seq)]
        scores.masked_fill_(attion_mask, -1e9)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, value).squeeze()
        return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hidden: int, n_heads: int, dropout: float = 0) -> None:
        super().__init__()
        assert d_hidden // n_heads != 0
        head_dim = int(d_hidden / n_heads)
        self.weight_q = nn.Linear(d_hidden, d_hidden)
        self.weight_k = nn.Linear(d_hidden, d_hidden)
        self.weight_v = nn.Linear(d_hidden, d_hidden)

        self.self_attention = ScaledDotProductAttention(
            head_dim=head_dim, dropout_rate=dropout
        )
        self.linear = nn.Linear(d_hidden, d_hidden)

        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor):
        """MultiHeadAttention

        Args:
            query (Tensor):  input word vector (bs, max_seq_len, d_hidden)
            key (Tensor): input word vector (bs, max_seq_len, d_hidden)
            value (Tensor): input word vector (bs, max_seq_len, d_hidden)
            attn_mask (Tensor): attn_mask (bs, max_seq_len, max_seq_len)

        Returns:
            _type_: _description_
        """
        batch_size = query.size(0)

        q_s = (
            self.weight_q(query)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        k_s = (
            self.weight_k(key)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        v_s = (
            self.weight_v(value)
            .view(batch_size, -1, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # => [batch_size, n_heads, len_q, head_dim]

        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # => [batch_size, n_heads, len_q, len_k]

        context, _ = self.self_attention(
            q_s, k_s, v_s, attn_mask
        )  # => [bs, n_heads, max_seq_size, d_hidden]

        context = (
            context.transpose(1, 2)  # => [bs, max_seq_size, n_heads, d_hidden]
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.head_dim)
        )  # => [batch_size, len_q, n_heads * head_dim]

        output = self.linear(context)  # => [batch_size, len_q, d_hidden]
        output = self.dropout(output)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_hidden, ff_dim)
        self.layer2 = nn.Linear(ff_dim, d_hidden)

        self.active = F.gelu  # gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.dropout(self.active(self.layer1(inputs)))
        output = self.dropout(self.layer2(output))
        return output


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        dropout: float = 0.0,
        layer_norm_epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.masked_mh = MultiHeadAttention(
            d_hidden=d_hidden, n_heads=n_heads, head_dim=head_dim, dropout=dropout
        )
        self.layer_norm_1 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)
        self.ffnn = PoswiseFeedForwardNet(
            d_hidden=d_hidden, ff_dim=ff_dim, dropout=dropout
        )
        self.layer_norm_2 = nn.LayerNorm(d_hidden, eps=layer_norm_epsilon)

    def forward(
        self,
        dec_input: Tensor,
        dec_self_attn_mask: Tensor,
    ) -> Tensor:
        masked_mh_ouput = self.masked_mh(
            dec_input, dec_input, dec_input, dec_self_attn_mask
        )
        masked_mh_ouput = self.layer_norm_1(masked_mh_ouput + dec_input)

        ffnn_output = self.ffnn(masked_mh_ouput)
        ffnn_output = self.layer_norm_2(ffnn_output + masked_mh_ouput)
        return ffnn_output


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        ff_dim: int,
        max_sequence_size: int,
        padding_id: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(input_dim, d_hidden)
        pe_table = get_position_encoding_table(max_sequence_size, d_hidden)
        self.pos_emb = nn.Embedding.from_pretrained(pe_table, freeze=True)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.padding_id = padding_id

    def forward(self, dec_inputs: Tensor):
        position = get_position(inputs=dec_inputs)
        conb_emb = self.src_emb(dec_inputs) + self.pos_emb(
            position
        )  # Embedding + pos_enbeding : [batch_size, max_seq_size, d_hidden]
        padding_mask = get_padding_mask(
            dec_inputs, self.padding_id
        )  # =>[batch_size, max_seq_size, max_seq_size]

        look_ahead_mask = get_look_ahead_mask(dec_inputs)

        dec_self_attn_mask = (
            padding_mask + look_ahead_mask
        )  # decoder 1번째 attention에 들어가는 mask

        dec_outputs = conb_emb
        for layer in self.layers:
            dec_ouputs = layer(
                dec_outputs, dec_self_attn_mask
            )  # => [bs, max_seq_size, d_hidden]

        return dec_ouputs


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_hidden: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        max_sequence_size: int,
        padding_id: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.decoder = Decoder(
            input_dim=vocab_size,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            max_sequence_size=max_sequence_size,
            padding_id=padding_id,
            dropout=dropout_rate,
        )

    def forward(self, dec_inputs: Tensor) -> Tensor:
        return self.decoder(dec_inputs)
