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


def test(task, model, language_encoder, n_episodes, n_steps_per_episode, device, environment, variations):
    model = model.to(device)
    model.eval()
    task = environment.get_task(hydra.utils.get_class(task))

    success = 0
    rgbs = []
    for i_episode in range(n_episodes):
        variation = np.random.choice(variations)
        task.set_variation(variation)
        if i_episode < 10:
            rgbs.append([])
        descriptions, obs = task.reset()
        description = descriptions[0]
        task_embed = language_encoder.encode([description])[0]
        task_embed = torch.from_numpy(task_embed).to(device)
        for j in range(n_steps_per_episode):
            if i_episode < 10:
                rgbs[-1].append(obs.front_rgb.copy().transpose(2, 0, 1))
            # image = Image.fromarray(obs.front_rgb)
            # plt.imshow(image)
            # plt.show()
            with torch.no_grad():
                action = model.get_action(obs, task_embed)
            try:
                obs, reward, terminate = task.step(action)
            except rlbench.backend.exceptions.InvalidActionError:
                terminate = True
            if j == n_steps_per_episode - 1:
                terminate = True
            if terminate:
                if reward == 1:
                    print('Success')
                    if i_episode < 10:
                        success_frame = np.zeros(obs.front_rgb.transpose(2, 0, 1).shape)
                        success_frame[2, :, :] = 255
                        rgbs[-1].append(success_frame)
                    success += 1
                else:
                    print('Failure')
                    if i_episode < 10:
                        failure_frame = np.zeros(obs.front_rgb.transpose(2, 0, 1).shape)
                        failure_frame[0, :, :] = 255
                        rgbs[-1].append(failure_frame)
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