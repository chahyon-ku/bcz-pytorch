import time
from matplotlib import pyplot as plt
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
from hydra.core.hydra_config import HydraConfig
os.environ['TOKENIZERS_PARALLELISM']='true'


@hydra.main(version_base=None, config_path='./configs/train/single_task', config_name='reach_target_carl')
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().run.dir
    input(cfg)

    logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.ERROR)
    task_embeds = {'reach_target': []}
    language_encoder = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens', cache_folder='cache')
    for variation in range(13):
        color_name, _ = colors[variation]
        languages = ['reach the %s target' % color_name,
                        'touch the %s ball with the panda gripper' % color_name,
                        'reach the %s sphere' % color_name]
        language = 'reach the %s target' % color_name
        task_embed = language_encoder.encode([language])[0]
        task_embeds['reach_target'].append(task_embed)

    environment = hydra.utils.instantiate(cfg.environment)
    train_dataset = hydra.utils.instantiate(cfg.train_dataset, task_embeds=task_embeds)
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader, dataset=train_dataset)
    train_dataloader_iter = iter(train_dataloader)
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.HuberLoss()

    # configure logging
    wandb.login()
    run = wandb.init(
        dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=os.path.basename(output_dir),
        **cfg.wandb
    )

    model.to(cfg.train.device)
    step_tqdm = tqdm.tqdm(range(1, cfg.train.n_steps + 1))
    train_losses = {}
    best_val_loss = float('inf')
    max_y = -1
    min_y = 1
    for step in step_tqdm:
        model.train()
        if step == 6000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if step == 9000:
            for g in optimizer.param_groups:
                g['lr'] = 5e-5
        try:
            image, task_embed, xyz, axangle, gripper = next(train_dataloader_iter)
        except StopIteration:
            print('restarting train dataloader')
            train_dataloader_iter = iter(train_dataloader)
            image, task_embed, xyz, axangle, gripper = next(train_dataloader_iter)
        model.train()
        image = image.to(cfg.train.device)
        action = xyz.to(cfg.train.device)
        # if step <= 10:
        #     print(np.linalg.norm(xyz, axis=-1), np.linalg.norm(axangle, axis=-1), gripper)
        #     plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        #     plt.title('xyz' + str(action[0, 0, :].cpu().numpy()))
        #     plt.show()
        task_embed = task_embed.to(cfg.train.device)
        optimizer.zero_grad()
        pred_action = model.forward(image)
        # input(pred_action[1].shape)
        # input(action[1].shape)
        # scale action y z by 10
        loss = loss_fn(pred_action, 20*action[:, 0, 1:])*200
        # print('pred_action', pred_action[0].detach().cpu().numpy())
        # print('action', 20*action[0, 0, 1:].detach().cpu().numpy())
        # y = pred_action[0].detach().cpu().numpy()[0]
        # if y > max_y:
        #     max_y = y
        #     print('max_y', max_y)
        # if y < min_y:
        #     min_y = y
        #     print('min_y', min_y)
            
        loss.backward()
        optimizer.step()

        # for k, v in loss.items():
        #     train_losses.setdefault(k, []).append(v.item())
        
        # if step % cfg.train.log_interval == 0:
        #     for k, v in train_losses.items():
        #         wandb.log({'train/loss/%s' % k: np.mean(v)}, step=step)
        #     train_losses = {}

        wandb.log({'train/loss/':loss.item()}, step=step)

        if step % cfg.train.eval_interval == 0:
            success_rate, rgbs = hydra.utils.instantiate(cfg.test, model=model, language_encoder=language_encoder, environment=environment)
            wandb.log({'test/success_rate': success_rate}, step=step)
            wandb.log({'test/rgb': [wandb.Video(np.stack(rgb, axis=0), fps=10) for rgb in rgbs]}, step=step)
            
        
        if step % cfg.train.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_{step}.pt'))
        

if __name__ == '__main__':
    main()