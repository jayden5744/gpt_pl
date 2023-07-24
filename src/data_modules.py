import multiprocessing

import lightning.pytorch as pl
import sentencepiece as spm
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

from src.datasets import GPTPretrainDataset, NaverClassificationDataset


class AbstractDataModule(pl.LightningDataModule):
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
        self.max_seq_size = arg_model.max_seq_size
        self.batch_size = batch_size
        self.train_dataset, self.valid_dataset = None, None

    def prepare_data(self) -> None:
        # 데이터를 다운로드, split 하거나 기타 등등
        # only called on 1 GPU/TPU in distributed
        return super().prepare_data()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_sampler = RandomSampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        valid_sampler = RandomSampler(self.valid_dataset)
        return DataLoader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count(),
            shuffle=False,  # validation에서는 shuffle 하지 않는 것은 권장함
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def teardown(self, stage: str) -> None:
        # clean up after fit or test
        # called on every process in DDP
        # setup 정반대
        return super().teardown(stage)


class PretrainDataModule(AbstractDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        arg_model: DictConfig,
        vocab: spm.SentencePieceProcessor,
        batch_size: int,
    ) -> None:
        super().__init__(arg_data, arg_model, vocab, batch_size)

    def setup(self, stage: str) -> None:
        # make assignments here (train/val/test split)
        # called on every process in DDP(distributed data parallel)
        self.train_dataset = GPTPretrainDataset(
            file_path=self.arg_data.train_path,
            vocab=self.vocab,
            max_seq_size=self.max_seq_size,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
            delim_token=self.arg_model.delim_token,
        )

        self.valid_dataset = GPTPretrainDataset(
            file_path=self.arg_data.valid_path,
            vocab=self.vocab,
            max_seq_size=self.max_seq_size,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
            delim_token=self.arg_model.delim_token,
        )


class NaverClassificationDataModule(AbstractDataModule):
    def __init__(
        self,
        arg_data: DictConfig,
        arg_model: DictConfig,
        vocab: spm.SentencePieceProcessor,
        batch_size: int,
    ) -> None:
        super().__init__(arg_data, arg_model, vocab, batch_size)

    def setup(self, stage: str) -> None:
        # make assignments here (train/val/test split)
        # called on every process in DDP(distributed data parallel)
        self.train_dataset = NaverClassifcationDataset(
            file_path=self.arg_data.train_path,
            vocab=self.vocab,
            max_seq_size=self.max_seq_size,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
            delim_token=self.arg_model.delim_token,
        )

        self.valid_dataset = NaverClassifcationDataset(
            file_path=self.arg_data.valid_path,
            vocab=self.vocab,
            max_seq_size=self.max_seq_size,
            bos_token=self.arg_model.bos_token,
            eos_token=self.arg_model.eos_token,
            unk_token=self.arg_model.unk_token,
            pad_token=self.arg_model.pad_token,
            delim_token=self.arg_model.delim_token,
        )
