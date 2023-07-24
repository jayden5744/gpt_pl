from abc import abstractmethod
from typing import Dict, Tuple

import lightning.pytorch as pl
import sentencepiece as spm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Optimizer

from src.data_modules import NaverClassificationDataModule
from src.model import GPT
from src.tasks import GPTPretrain
from src.utils import create_or_load_tokenizer


class AbstractModule(pl.LightningModule):
    def __init__(self, arg: DictConfig) -> None:
        super().__init__()
        self.arg = arg
        self.vocab = self.get_vocab()
        self.pad_id = self.vocab.PieceToId(self.arg.model.pad_token)
        self.gpt = self.get_gpt_model()

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer_type = self.arg.trainer.optimizer
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.arg.trainer.learning_rate,
                betas=(self.arg.trainer.optimizer_b1, self.arg.trainer.optimizer_b2),
                eps=self.arg.trainer.optimizer_e,
                weight_decay=self.arg.trainer.weight_decay,
            )

        else:
            raise ValueError("trainer param `optimizer` must be one of [Adam, AdamW].")
        return optimizer

    def get_vocab(self) -> Tuple[spm.SentencePieceProcessor]:
        vocab = create_or_load_tokenizer(
            file_path=self.arg.data.train_path,
            save_path=self.arg.data.dictionary_path,
            language=self.arg.data.language,
            vocab_size=self.arg.data.vocab_size,
            tokenizer_type=self.arg.data.tokenizer_type,
            bos_token=self.arg.model.bos_token,
            eos_token=self.arg.model.eos_token,
            unk_token=self.arg.model.unk_token,
            pad_token=self.arg.model.pad_token,
        )
        return vocab

    def get_gpt_model(self) -> nn.Module:
        params = {
            "vocab_size": self.arg.data.vocab_size,
            "d_hidden": self.arg.model.d_hidden,
            "n_heads": self.arg.model.n_heads,
            "ff_dim": self.arg.model.d_hidden * 4,  # ff_dim은 d_hidden * 4이다(페이퍼)
            "n_layers": self.arg.model.n_layers,
            "max_sequence_size": self.arg.model.max_seq_size,
            "dropout_rate": self.arg.model.dropout_rate,
            "padding_id": self.pad_id,
        }
        return GPT(**params)


class GPTPretrainModule(AbstractModule):
    def __init__(self, arg: DictConfig) -> None:
        super().__init__(arg)
        self.model = self.get_model()

        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )

    def _shared_eval_step(self, batch, batch_idx: int) -> Tensor:
        # validation step과 test step의 공통으로 사용되는 부분
        dec_inputs = batch
        labels = dec_inputs[:, 1:].contiguous()
        output = self.model(dec_inputs)

        return self.calculate_loss(output, labels)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        dec_inputs = batch
        labels = dec_inputs[:, 1:].contiguous()
        output = self.model(dec_inputs)

        loss = self.calculate_loss(output, labels)

        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def on_validation_epoch_end(self):
        # validation 1 epoch 끝나고 나서 수행하게 될 로직
        pass

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        if self.device.type == "mps":
            # mps float64를 처리할 수 없음
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            output = output.to(device="cpu")
            target = target.to(device="cpu")
        return self.loss_function(output.view(-1, output.size(2)), target.view(-1))

    def get_model(self) -> nn.Module:
        return GPTPretrain(
            gpt_model=self.gpt,
            vocab_size=self.arg.data.vocab_size,
            d_hidden=self.arg.model.d_hidden,
        )


class NaverClassificationModule(AbstractModule):
    def __init__(self, arg: DictConfig) -> None:
        super().__init__(arg)
        self.model = self.get_model()
        self.pretrain_path = arg.data.pretrain_path
        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=self.arg.model.pad_id,
            label_smoothing=self.arg.trainer.label_smoothing_value,
        )

    def _shared_eval_step(self, batch, batch_idx: int) -> Tensor:
        # validation step과 test step의 공통으로 사용되는 부분
        dec_inputs, labels = batch
        output = self.model(dec_inputs)

        return self.calculate_loss(output, labels)

    def training_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        dec_inputs, labels = batch
        output = self.model(dec_inputs)

        loss = self.calculate_loss(output, labels)

        metrics = {"loss": loss}
        self.log_dict(metrics)
        return metrics

    def validation_step(self, batch, batch_idx: int) -> Dict[str, Tensor]:
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def on_validation_epoch_end(self):
        # validation 1 epoch 끝나고 나서 수행하게 될 로직
        pass

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        if self.device.type == "mps":
            # mps float64를 처리할 수 없음
            # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
            output = output.to(device="cpu")
            target = target.to(device="cpu")

        return self.loss_function(output, target)

    def get_model(self) -> nn.Module:
        self.gpt.load_state_dict(torch.load(self.pretrain_path)["state_dict"])
        return NaverClassificationDataModule(
            gpt_model=self.gpt,
            d_hidden=self.arg.model.d_hidden,
            n_outputs=self.arg.model.n_outputs,
        )
