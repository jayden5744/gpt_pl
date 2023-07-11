import os
from typing import Dict, Tuple
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from src.data_modules import PretrainDataModule, NaverClassificationDataModule

from src.train import GPTPretrainModel, NaverClassificationModel

def make_config(cfg: DictConfig) -> Dict:
    result = {}
    result.update(dict(cfg.data))
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))
    return result


def get_model_n_data_module(cfg) -> Tuple[pl.LightningModule, pl.LightningDataModule]:
    if cfg.data.task == "pretrain":
        model = GPTPretrainModel(arg=cfg)

        data_module = PretrainDataModule(
            arg_data=cfg.data,
            arg_model=cfg.model,
            vocab=model.vocab,
            batch_size=cfg.trainer.batch_size
        )

    elif cfg.data.task == "classification":
        model = NaverClassificationModel(arg=cfg)
        model.model.gpt.load(cfg.data.pretrain_path)

        data_module = NaverClassificationDataModule(
            arg_data=cfg.data,
            arg_model=cfg.model,
            vocab=model.vocab,
            batch_size=cfg.trainer.batch_size
        )
    else:
        raise ValueError()
    return model, data_module




@hydra.main(version_base="1.3.2", config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:

    model, data_module = get_model_n_data_module(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(get_original_cwd(), "./SavedModel/"),
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
    wandb_logger = WandbLogger(project=cfg.data.project_name, name=cfg.data.folder_name)
    wandb_logger.log_hyperparams(make_config(cfg))

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=cfg.trainer.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        # logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    train()