from abc import abstractmethod
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
import pandas as pd

import sentencepiece as spm


class AbstractDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        vocab: spm.SentencePieceProcessor,
        max_seq_size: int,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        delim_token: str = "$",
        ) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_seq_size = max_seq_size

        self.bos_id = vocab.PieceToId(bos_token)
        self.eos_id = vocab.PieceToId(eos_token)
        self.unk_id = vocab.PieceToId(unk_token)
        self.pad_id = vocab.PieceToId(pad_token)
        self.delim_id = vocab.PieceToId(delim_token)

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
        max_seq_size: int,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        delim_token: str = "$",
        ) -> None:
        super().__init__(file_path, vocab, max_seq_size, bos_token, eos_token, unk_token, pad_token, delim_token)
        self.data = self.load_txt_file(file_path)

    def __getitem__(self, index):
        doc = self.vocab.EncodeAsIds(self.data[index])
        if len(doc) + 2 > self.max_seq_size:
            doc = doc[:self.max_seq_size - 2]
        return torch.tensor([self.bos_id] + doc + [self.eos_id] + [self.pad_id] * (self.max_seq_size - 2 - len(doc)))

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
        max_seq_size: int,
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        delim_token: str = "$",
        ) -> None:
        super().__init__(file_path, vocab, max_seq_size, bos_token, eos_token, unk_token, pad_token, delim_token)
        self.sentences, self.labels =  self.load_txt_file(file_path)

    def __getitem__(self, index: int):
        try:
            doc = self.vocab.EncodeAsIds(self.sentences[index])
        except TypeError as e:
            print("sentences : ", self.sentences[index])
            raise e
        
        if len(doc) + 2 > self.max_seq_size:
            doc = doc[:self.max_seq_size - 2]
        return torch.tensor([self.bos_id] + doc + [self.eos_id] + [self.pad_id] * (self.max_seq_size - 2 - len(doc))), torch.tensor(self.labels[index])

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
            sentence = str(row["document"]).strip()
            try:
                self.vocab.EncodeAsIds(sentence)

            except TypeError as e:
                continue
            labels.append(row["label"])
            sentences.append(sentence)
        return sentences, labels

    


