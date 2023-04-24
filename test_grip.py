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
    model.eval()
    task = environment.get_task(hydra.utils.get_class(task))
    success = 0
    rgbs = []
    for i_episode in range(n_episodes):
        rgbs.append([])
        # if i_episode % 2 == 0:
        task.set_variation(0)
        # else:
        #     task.set_variation(2)
        descriptions, obs = task.reset()
        description = descriptions[0]
        task_embed = language_encoder.encode([description])[0]
        task_embed = torch.from_numpy(task_embed).to(device)
        grippers = []
        closed = False
        for j in range(n_steps_per_episode):
            rgbs[-1].append(obs.front_rgb.copy().transpose(2, 0, 1))
            # image = Image.fromarray(obs.front_rgb)
            # plt.imshow(image)
            # plt.show()
            action = model.get_action(obs, task_embed)
            # if action[-1] > 0.5:
            #     action[-1] = 1
            # else:
            #     action[-1] = 0
            # grippers.append(action[-1])
            # if grippers[-1] == 0:
            #     closed = True
            # # once gripper closes, keep it closed for n steps
            # n = 10
            # if len(grippers) > n and closed and grippers[-n] == 1:
            #     action[-1] = 0
            #     grippers[-1] = 0

            # if grippers[-1] == 1:
            #     closed = False
            #     # set whole list to 1
            #     for i in range(len(grippers)):
            #         grippers[i] = 1
            
            # # at certain steps, close gripper
            # if j > 70 and j < 120:
            #     action[-1] = 0
            # # at certain steps, open gripper
            # if j > 140:
            #     action[-1] = 1

            # # if previous 3 grippers are the same, allow change
            # if len(grippers) > 4 and grippers[-2] == grippers[-3] == grippers[-4]:
            #     pass
            # elif len(grippers) > 1:
            #     action[-1] = grippers[-2]
            #     grippers[-1] = grippers[-2]
            # print(action)
            # print(gripper)
            try:
                obs, reward, terminate = task.step(action)
            except rlbench.backend.exceptions.InvalidActionError:
                print('Invalid action')
                break
            if terminate:
                if reward == 1:
                    print('Success')
                    success += 1
                # do 10 more steps then break
                for k in range(10):
                    rgbs[-1].append(obs.front_rgb.copy().transpose(2, 0, 1))
                    action = model.get_action(obs, task_embed)
                    try:
                        obs, reward, terminate = task.step(action)
                    except rlbench.backend.exceptions.InvalidActionError:
                        print('Invalid action')
                        break
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