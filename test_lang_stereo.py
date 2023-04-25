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
from PIL import Image, ImageDraw, ImageFont, ImageColor
os.environ['TOKENIZERS_PARALLELISM']='true'
import clip

def test(task, model, language_encoder, n_episodes, n_steps_per_episode, device, environment):
    model = model.to(device)
    model.eval()
    task = environment.get_task(hydra.utils.get_class(task))
    success = 0
    rgbs = []
    for i_episode in range(n_episodes):
        rgbs.append([])
        task.set_variation(i_episode % 4)
        # task.set_variation(7)

        descriptions, obs = task.reset()
        description = descriptions[0]
        # tokenize description
        text = clip.tokenize([description]).to(device)
        task_embed = language_encoder.encode_text(text).float()
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
            # check if it's the last step
            if j == n_steps_per_episode - 1:
                terminate = True
            if terminate:
                if reward == 1:
                    print('Success')
                    success += 1
                else:
                    print('Failure')
                # do 10 more steps then break
                for k in range(10):
                    rgbs[-1].append(obs.front_rgb.copy().transpose(2, 0, 1))
                    action = model.get_action(obs, task_embed, zero=True)
                    try:
                        obs, _, terminate = task.step(action)
                    except rlbench.backend.exceptions.InvalidActionError:
                        break
                break
        # if it was success, append green image, else red
        if reward == 1:
            # create green image using PIL
            im = Image.new('RGB', (128, 128), color='green')
        else:
            # create red image using PIL
            im = Image.new('RGB', (128, 128), color='red')
        # draw text on image
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype('arial.ttf', 20)
        draw.text((10, 10), description, font=font, fill='black')
        # cast to np array uint8
        im = np.array(im)
        # cast to uint8
        im = im.astype(np.uint8)
        for _ in range(10):
            rgbs[-1].append(im)
        print('success rate: ', success / (i_episode + 1), i_episode + 1)
    print('final success rate: ', success / n_episodes)
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