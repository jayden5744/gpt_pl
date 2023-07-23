import os
from typing import Dict, Tuple

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import get_original_cwd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from src.callbacks import SelectedLayerCheckpoint

from src.data_modules import NaverClassificationDataModule, PretrainDataModule
from src.train import GPTPretrainModule, NaverClassificationModule


def make_config(cfg: DictConfig) -> Dict:
    result = {}
    result.update(dict(cfg.data))
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))
    return result


def get_model_n_data_module(cfg) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    if cfg.data.task == "pretrain":
        module = GPTPretrainModule(arg=cfg)
        data_module = PretrainDataModule(
            arg_data=cfg.data,
            arg_model=cfg.model,
            vocab=module.vocab,
            batch_size=cfg.trainer.batch_size,
        )

    elif cfg.data.task == "classification":
        module = NaverClassificationModule(arg=cfg)
        data_module = NaverClassificationDataModule(
            arg_data=cfg.data,
            arg_model=cfg.model,
            vocab=module.vocab,
            batch_size=cfg.trainer.batch_size,
        )

    else:
        raise ValueError
    return module, data_module


@hydra.main(version_base="1.3.2", config_path="configs", config_name="classification")
def train(cfg: DictConfig) -> None:
    callback_lst = []
    module, data_module = get_model_n_data_module(cfg)

    if cfg.data.task == "pretrain":
        selected_layer_checkpoint_callback = SelectedLayerCheckpoint()
        callback_lst.append(selected_layer_checkpoint_callback)


    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            get_original_cwd(), f"./SavedModel/{cfg.data.folder_name}"
        ),
        filename=cfg.data.folder_name,
        save_top_k=5,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.trainer.early_stopping,
        verbose=False,
        mode="min",
    )
    callback_lst.append(checkpoint_callback, early_stop_callback)
    # wandb_logger = WandbLogger(project=cfg.data.project_name, name=cfg.data.folder_name)
    # wandb_logger.log_hyperparams(make_config(cfg))

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=cfg.trainer.epochs,
        callbacks=callback_lst,
        strategy="ddp_find_unused_parameters_true"
        # logger=wandb_logger,
    )
    trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    train()
