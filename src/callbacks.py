import torch
import torch.nn as nn
import pytorch_lightning as pl


class SelectedLayerCheckpoint(pl.Callback):
    def __init__(self):
        self.selected_layers = ["model.gpt"]

    def on_epoch_end(self, trainer, pl_module):
        state_dict = {}
        for name, param in pl_module.named_parameters():
            if any(layer_name in name for layer_name in self.selected_layers):
                state_dict[name] = param
        torch.save(state_dict, f'{trainer.default_root_dir}/selected_layers_checkpoint.pth')
