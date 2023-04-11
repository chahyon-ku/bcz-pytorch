import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.xyz_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 30),
        )
        self.axangle_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 30),
        )
        self.gripper_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid(),
        )
    
    def forward(self, image_embed):
        # image_embed: (batch_size, 512)
        xyz = self.xyz_branch(image_embed)
        axangle = self.axangle_branch(image_embed)
        gripper = self.gripper_branch(image_embed)

        xyz = torch.reshape(xyz, (-1, 10, 3))
        axangle = torch.reshape(axangle, (-1, 10, 3))
        gripper = torch.reshape(gripper, (-1, 10, 1))
        # xyz: (batch_size, 10, 3)
        # axangle: (batch_size, 10, 4)
        # gripper: (batch_size, 10, 1)
        return xyz, axangle, gripper
