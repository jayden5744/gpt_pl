import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    train()