import time
from matplotlib import pyplot as plt
# import rlbench.utils
# import rlbench.action_modes.action_mode
# import rlbench.action_modes.arm_action_modes
# import rlbench.action_modes.gripper_action_modes
# import rlbench.observation_config
# import rlbench.tasks.reach_target
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
import clip

@hydra.main(version_base=None, config_path='./configs/train/single_task', config_name='push_block')
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().run.dir
    input(cfg)

    # create clip model
    device = 'cuda:1'
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    # create list of text descriptions
    task_embeds = {'reach_target': []}
    text = []
    text.append('push the magenta block left')
    text.append('push the cyan block forward')
    text.append('push the magenta block forward')
    text.append('push the cyan block left')
    text.append('push the yellow block left')
    text.append('push the yellow block forward')
    text.append('push the lime block left')
    # get embeddings
    tokens = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    # put in dictionary
    for i in range(len(text)):
        task_embeds['reach_target'].append(text_features[i].float().detach().cpu().numpy())

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
    train_losses = []
    best_val_loss = float('inf')
    for step in step_tqdm:
        model.train()
        if step == 7000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if step == 10000:
            for g in optimizer.param_groups:
                g['lr'] = 5e-5
        try:
            image, image2, task_embed, xyz, axangle, gripper, curr_gripper = next(train_dataloader_iter)
        except StopIteration:
            print('restarting train dataloader')
            train_dataloader_iter = iter(train_dataloader)
            image, image2, task_embed, xyz, axangle, gripper, curr_gripper = next(train_dataloader_iter)
        model.train()
        image = image.to(cfg.train.device)
        image2 = image2.to(cfg.train.device)
        action = xyz.to(cfg.train.device)
        # if step <= 10:
        #     print(np.linalg.norm(xyz, axis=-1), np.linalg.norm(axangle, axis=-1), gripper)
        #     plt.imshow(image[0].permute(1, 2, 0).cpu().numpy())
        #     plt.title('xyz' + str(action[0, 0, :].cpu().numpy()))
        #     plt.show()
        task_embed = task_embed.to(cfg.train.device)
        optimizer.zero_grad()
        pred_action = model.forward(image, image2, task_embed)
        # input(pred_action[1].shape)
        # input(action[1].shape)
        # scale action y z by 10
        action_scale = 40
        loss = loss_fn(pred_action, action_scale*action[:, 0, :])*100
        # if step % 100 == 0:
        #     print('pred_action', pred_action[0].detach().cpu().numpy())
        #     print('action', action_scale*action[0, 0, :].detach().cpu().numpy())
            
        loss.backward()
        optimizer.step()

        # for k, v in loss.items():
        #     train_losses.setdefault(k, []).append(v.item())
        
        # if step % cfg.train.log_interval == 0:
        #     for k, v in train_losses.items():
        #         wandb.log({'train/loss/%s' % k: np.mean(v)}, step=step)
        #     train_losses = {}

        train_losses.append(loss.item())
        if step % cfg.train.log_interval == 0:
            wandb.log({'train/loss/':np.mean(train_losses)}, step=step)
            train_losses = []

        if step % cfg.train.eval_interval == 0:
            success_rate, rgbs = hydra.utils.instantiate(cfg.test, model=model, language_encoder=clip_model, environment=environment)
            wandb.log({'test/success_rate': success_rate}, step=step)
            wandb.log({'test/rgb': [wandb.Video(np.stack(rgb, axis=0), fps=10) for rgb in rgbs]}, step=step)
            
        
        if step % cfg.train.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_{step}.pt'))
        

if __name__ == '__main__':
    main()