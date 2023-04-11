import time
import rlbench.utils
import rlbench.action_modes.action_mode
import rlbench.action_modes.arm_action_modes
import rlbench.action_modes.gripper_action_modes
import rlbench.observation_config
import rlbench.tasks.reach_target
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sentence_transformers
from rlbench.const import colors
import torchvision
import tqdm
import os
import logging
import wandb
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from PIL import Image
os.environ['TOKENIZERS_PARALLELISM']='true'


def test(task, model, language_encoder, n_episodes, n_steps_per_episode, device, environment):
    model = model.to(device)
    task = environment.get_task(hydra.utils.get_class(task))

    success = 0
    rgbs = []
    for i_episode in range(n_episodes):
        rgbs.append([])
        descriptions, obs = task.reset()
        description = descriptions[0]
        task_embed = language_encoder.encode([description])[0]
        task_embed = torch.from_numpy(task_embed).to(device)
        for j in range(n_steps_per_episode):
            rgbs[-1].append(obs.front_rgb.copy().transpose(2, 0, 1))
            # image = Image.fromarray(obs.front_rgb)
            # plt.imshow(image)
            # plt.show()
            action = model.get_action(obs, task_embed)
            try:
                obs, reward, terminate = task.step(action)
            except rlbench.backend.exceptions.InvalidActionError:
                print('Invalid action')
                break
            if terminate:
                if reward == 1:
                    print('Success')
                    success += 1
                break

    # environment.shutdown()
    return success / n_episodes, rgbs


@hydra.main(version_base=None, config_path='./configs/test', config_name='reach_target')
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().run.dir

    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.ERROR)
    language_encoder = hydra.utils.instantiate(cfg.language_encoder)
    model = hydra.utils.instantiate(cfg.model)

    hydra.utils.instantiate(cfg.test, model=model, language_encoder=language_encoder)
        

if __name__ == '__main__':
    main()