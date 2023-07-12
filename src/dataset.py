
from typing import List

import torch
from torch import Tensor
import sentencepiece as spm
from torch.utils.data import Dataset


class GPTPretrainDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_size: int,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]"
        ) -> None:
        super().__init__()
        self.data = self.load_txt_file(file_path)
        self.vocab = vocab
        self.max_seq_size = max_seq_size

        self.bos_id = vocab.PieceToId(bos_token)
        self.eos_id = vocab.PieceToId(eos_token)
        self.unk_id = vocab.PieceToId(unk_token)
        self.pad_id = vocab.PieceToId(pad_token)


    def __getitem__(self, index) -> Tensor:
        doc = self.vocab.EncodeAsIds(self.data[index])
        if len(doc) + 2 > self.max_seq_size:
            doc = doc[:self.max_seq_size - 2]
        return torch.tensor([self.bos_id] + doc + [self.eos_id] + [self.pad_id] * (self.max_seq_size - 2 - len(doc)))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
      for x in self.data:
        yield x

    def get_vocab(self) -> spm.SentencePieceProcessor:
        return self.vocab

    def decode(self, x: str) -> str:
        return self.vocab.DecodeIds(x)

    def load_txt_file(self, file_path: str) -> List[str]:
        dataset = []
        with open(file_path, 'r') as f:
            for data in f.readlines():
                dataset.append(data.strip())
        return dataset
