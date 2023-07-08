import multiprocessing
import os
import os.path as osp
import shutil
from typing import List

import torch
from torch import Tensor
import sentencepiece as spm
from omegaconf import DictConfig
from torch.utils.data import Dataset,  DataLoader, RandomSampler

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


def exist_file(path: str) -> bool:
    if osp.exists(path):
        return True
    return False


def create_or_load_tokenizer(
    file_path: str,
    save_path: str,
    language: str,
    vocab_size: int,
    tokenizer_type: str = "bpe",
    bos_token: str = "[BOS]",
    eos_token: str = "[EOS]",
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]"
) -> spm.SentencePieceProcessor:
    corpus_prefix = f"{language}_corpus_{vocab_size}"

    if tokenizer_type.strip().lower() not in ["unigram", "bpe", "char", "word"]:
        raise ValueError(
            f"param `tokenizer_type` must be one of [unigram, bpe, char, word]"
        )

    if not os.path.isdir(save_path):  # 폴더 없으면 만들어
        os.makedirs(save_path)

    model_path = osp.join(save_path, corpus_prefix + ".model")
    vocab_path = osp.join(save_path, corpus_prefix + ".vocab")

    if not exist_file(model_path) and not exist_file(vocab_path):
        model_train_cmd = f"--input={file_path} --model_prefix={corpus_prefix} --model_type={tokenizer_type} --vocab_size={vocab_size}  --bos_piece={bos_token}  --eos_piece={eos_token}  --unk_piece={unk_token} --pad_piece={pad_token}"
        spm.SentencePieceTrainer.Train(model_train_cmd)
        shutil.move(corpus_prefix + ".model", model_path)
        shutil.move(corpus_prefix + ".vocab", vocab_path)
    # model file은 있는데, vocab file이 없거나 / model_file은 없는데, vocab file이 있으면 -> Error

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


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


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        arg_model: DictConfig,
        vocab: spm.SentencePieceProcessor,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.arg_data = arg_data
        self.arg_model = arg_model
        self.vocab = vocab
        self.max_seq_len = arg_model.max_seq_len
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # 데이터를 다운로드, split 하거나 기타 등등
        # only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        # make assignments here (train/val/test split)
        # called on every process in DDP(distributed data parallel)
        self.train_dataset = GPTPretrainDataset(
            file_path=self.arg_data.train_path,
            vocab=self.vocab,
            max_seq_len=self.max_seq_len,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
        )

        self.valid_dataset = GPTPretrainDataset(
            file_path=self.arg_data.valid_path,
            vocab=self.vocab,
            max_seq_len=self.max_seq_len,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count()
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False  # validation에서는 shuffle 하지 않는 것은 권장함
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:
        # clean up after fit or test
        # called on every process in DDP
        # setup 정반대
        return super().teardown(stage)