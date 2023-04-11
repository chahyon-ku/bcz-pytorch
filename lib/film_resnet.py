'''
https://github.com/pytorch/vision/blob/v0.14.1/torchvision/models/resnet.py
'''

import torch
import torchvision
import torch.nn as nn
from typing import List, Type, Any, Callable, Union, Optional, Tuple


class FilmGenerator(nn.Module):
    def __init__(self, task_embed_dim) -> None:
        super(FilmGenerator, self).__init__()
        self.fc = nn.Linear(task_embed_dim, 2 * (128 + 256 + 512 + 1024))

    def forward(self, task_embed: torch.Tensor) -> torch.Tensor:
        out = self.fc(task_embed)
        gamma = [out[:, :128, None, None],
                 out[:, 128:384, None, None],
                 out[:, 384:896, None, None],
                 out[:, 896:1920, None, None]]
        beta = [out[:, 1920:2048, None, None],
                out[:, 2048:2304, None, None],
                out[:, 2304:2816, None, None],
                out[:, 2816:3840, None, None]]

        return gamma, beta


class FilmBlock(torchvision.models.resnet.BasicBlock):
    def __init__(self, *args, **kwargs) -> None:
        super(FilmBlock, self).__init__(*args, **kwargs)

    def forward(self, x) -> torch.Tensor:
        x, gamma, beta = x
        # identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        identity = out

        out = self.conv2(out)
        out = self.bn2(out)
        gamma, new_gamma = gamma[:, :out.shape[1]], gamma[:, out.shape[1]:]
        beta, new_beta = beta[:, :out.shape[1]], beta[:, out.shape[1]:]
        # out = gamma * out + beta

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, new_gamma, new_beta


class FilmResnet(torchvision.models.resnet.ResNet):
    def __init__(self, task_embed_dim, **kwargs) -> None:
        super(FilmResnet, self).__init__(**kwargs)
        self.film_generator = FilmGenerator(task_embed_dim)

    def _forward_impl(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        gamma, beta = self.film_generator(task_embed)
        x, _, _ = self.layer1((x, gamma[0], beta[0]))
        x, _, _ = self.layer2((x, gamma[1], beta[1]))
        x, _, _ = self.layer3((x, gamma[2], beta[2]))
        x, _, _ = self.layer4((x, gamma[3], beta[3]))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
    
    def forward(self, x: torch.Tensor, task_embed: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, task_embed)


def film_resnet18(task_embed_dim):
    return FilmResnet(
        task_embed_dim=task_embed_dim,
        block=FilmBlock,
        layers=[2, 2, 2, 2],
    )

def film_resnet34(task_embed_dim):
    return FilmResnet(
        task_embed_dim=task_embed_dim,
        block=FilmBlock,
        layers=[3, 4, 6, 3],
    )