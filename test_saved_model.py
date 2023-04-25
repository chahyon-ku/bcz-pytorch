import torch
from lib.model_lang_stereo import BC
import clip
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb
import os
import numpy as np

@hydra.main(version_base=None, config_path='./configs/train/single_task', config_name='push_block')
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().run.dir
    input(cfg)
    device = 'cuda:1'
    # initialize model
    model = BC()
    # load model
    model.load_state_dict(torch.load('models/model_push_block_60.pt'))
    model.eval()
    model.to(device)

    # initialize clip model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    environment = hydra.utils.instantiate(cfg.environment)

    # configure logging
    wandb.login()
    run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    # run test
    step = 0
    success_rate, rgbs = hydra.utils.instantiate(cfg.test, model=model, language_encoder=clip_model, environment=environment)
    wandb.log({'test/success_rate': success_rate}, step=step)
    wandb.log({'test/rgb': [wandb.Video(np.stack(rgb, axis=0), fps=10) for rgb in rgbs]}, step=step)

if __name__ == '__main__':
    main()