import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import RelLossConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="rel_loss", node=RelLossConfig)


@hydra.main(config_name='rel_loss', config_path='configs/docred')
def rel_loss(cfg: RelLossConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.dataset, 'train_path')
    util.config_to_abs_paths(cfg.loss, 'output_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')

    model.rel_loss(cfg)


if __name__ == '__main__':
    rel_loss()
