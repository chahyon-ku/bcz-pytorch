import glob
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
                reward = 0
            if j == n_steps_per_episode - 1:
                terminate = True
                reward = 0
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

    environment = hydra.utils.instantiate(cfg.environment)
    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.ERROR)
    language_encoder = hydra.utils.instantiate(cfg.language_encoder)
    model = hydra.utils.instantiate(cfg.model)

    model_paths = glob.glob(cfg.model_glob)
    model_paths = sorted(model_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # model_paths = list(filter(lambda x: (int(x.split('_')[-1].split('.')[0]) % 5000) == 0, model_paths))
    
    print(model_paths, cfg.model_glob)
    wandb.login()
    run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    for model_path in model_paths:
        for variation in tqdm.tqdm(cfg.test.variations):
            model.load_state_dict(torch.load(model_path))
            success_rate, rgbs = hydra.utils.instantiate(cfg.test, model=model, language_encoder=language_encoder, variations=[variation], environment=environment)
            rgbs = rgbs[:10]
            step = variation#int(model_path.split('_')[-1].split('.')[0])
            wandb.log({'test/success_rate': success_rate}, step=step)
            wandb.log({'test/rgb': [wandb.Video(np.stack(rgb, axis=0), fps=10) for rgb in rgbs]}, step=step)

if __name__ == '__main__':
    main()