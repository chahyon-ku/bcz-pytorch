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
os.environ['TOKENIZERS_PARALLELISM']='true'


@hydra.main(version_base=None, config_path='./configs/train/single_task', config_name='reach_target')
def main(cfg: DictConfig) -> None:
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
    print([task_embed.shape for task_embed in task_embeds['reach_target']])
    del language_encoder

    train_dataset = hydra.utils.instantiate(cfg.train_dataset, task_embeds=task_embeds)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset, task_embeds=task_embeds)
    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader, dataset=train_dataset)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader, dataset=val_dataset)
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    # scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    # loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    model.to('cuda:1')
    for epoch in range(cfg.train.epochs):
        model.train()
        train_tqdm = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Epoch %d' % epoch)
        for i_batch, (image, task_embed, action) in train_tqdm:
            image = image.to('cuda:1')
            task_embed = task_embed.to('cuda:1')
            action = action.to('cuda:1')
            optimizer.zero_grad()
            pred_action = model(image, task_embed)
            # loss = loss_fn(output, batch)
            # loss.backward()
            optimizer.step()
        # scheduler.step()

        model.eval()
        with torch.no_grad():
            val_tqdm = tqdm.tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Epoch %d' % epoch)
            for i_batch, (image, task_embed, action) in val_tqdm:
                image = image.to('cuda:1')
                task_embed = task_embed.to('cuda:1')
                action = action.to('cuda:1')
                output = model(image, task_embed)
                # loss = loss_fn(output, batch)
    


if __name__ == '__main__':
    main()