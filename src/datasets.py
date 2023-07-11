from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd

import sentencepiece as spm


class AbstractDataset(Dataset, metaclass=ABC):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_len: int,
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        pad_id: int = 3,
        ) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_seq_len = max_seq_len

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.pad_id = pad_id

    def get_vocab(self) -> spm.SentencePieceProcessor:
        return self.vocab

    def decode(self, x: str) -> str:
        return self.vocab.DecodeIds(x)

    @abstractmethod
    def load_txt_file(self, file_path: str):
        pass


class GPTPretrainDataset(AbstractDataset):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_len: int,
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        pad_id: int = 3,
        ) -> None:
        super(AbstractDataset).__init__(file_path, vocab, max_seq_len, bos_id, eos_id, unk_id, pad_id)
        self.data = self.load_txt_file(file_path)

    def __getitem__(self, index):
        doc = self.vocab.EncodeAsIds(self.data[index])
        if len(doc) + 2 > self.max_seq_len:
            doc = doc[:self.max_seq_len - 2]
        return torch.tensor([self.bos_id] + doc + [self.eos_id] + [self.pad_id] * (self.max_seq_len - 2 - len(doc)))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
      for x in self.data:
        yield x

    def load_txt_file(self, file_path: str) -> List[str]:
        dataset = []
        with open(file_path, 'r') as f:
            for data in f.readlines():
                dataset.append(data.strip())
        return dataset


class NaverClassificationDataset(AbstractDataset):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_len: int,
        bos_id: int = 0,
        eos_id: int = 1,
        unk_id: int = 2,
        pad_id: int = 3,
        ) -> None:
        super(AbstractDataset).__init__(file_path, vocab, max_seq_len, bos_id, eos_id, unk_id, pad_id)
        self.sentences, self.labels =  self.load_txt_file(file_path)

    def __getitem__(self, index: int):
        doc = self.vocab.EncodeAsIds(self.sentences[index])
        if len(doc) + 2 > self.max_seq_len:
            doc = doc[:self.max_seq_len - 2]
        return torch.tensor([self.bos_id] + doc + [self.eos_id] + [self.pad_id] * (self.max_seq_len - 2 - len(doc))), torch.tensor(self.labels[index])

    def __len__(self) -> int:
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)

    def __iter__(self):
      for x, y in zip(self.sentences, self.labels):
        yield x, y

    def load_txt_file(self, file_path: str) -> Tuple[List[str], List[str]]:
        df = pd.read_csv(file_path, sep="\t", engine="python")
        sentences = []
        labels = []
        for i, row in df.iterrows():
            labels.append(row["label"])
            sentences.append(row["document"])
        return sentences, labels



