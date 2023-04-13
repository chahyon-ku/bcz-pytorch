import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
from PIL import Image


class BCZModel(torch.nn.Module):
    def __init__(self, vision_encoders, policy_decoder, views, model_path=None) -> None:
        super().__init__()
        self.vision_encoders = vision_encoders
        self.policy_decoder = policy_decoder
        self.views = views
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, image, task_embed):
        B, O, V, C, H, W = image.shape
        image_embed = torch.concat([self.vision_encoders[i](image[:, :, i], task_embed) for i in range(V)], dim=-1)
        xyz, axangle, gripper = self.policy_decoder(image_embed)
        return xyz, axangle, gripper

    def get_action(self, obs, task_embed):
        # obs: rlbench.observation.Observation
        # task_embed: (768,)
        images = torch.stack([TF.to_tensor(Image.fromarray(obs.__dict__[view])) for view in self.views], dim=0)[None, None, ...].to(task_embed)
        task_embed = task_embed.unsqueeze(0)
        xyz, axangle, gripper = self.forward(images, task_embed)
        # xyz: (batch_size, 10, 3)
        # axangle: (batch_size, 10, 4)
        # gripper: (batch_size, 10, 1)
        xyz = xyz.detach().cpu().numpy()[0, 0]
        axangle = axangle.detach().cpu().numpy()[0, 0]
        gripper = gripper.detach().cpu().numpy()[0, 0]
        curr_xyz = obs.gripper_pose[:3]
        curr_quat = obs.gripper_pose[3:]
        curr_axangle = R.from_quat(curr_quat).as_rotvec()
        xyz = curr_xyz + xyz
        axangle = curr_axangle + axangle
        # axangle = curr_axangle
        quat = R.from_rotvec(axangle).as_quat()
        
        action = np.concatenate([xyz, quat, gripper])

        return action

    def loss(self, input, target):
        xyz, axangle, gripper = input
        xyz_target, axangle_target, gripper_target = target
        xyz_loss = 100 * torch.nn.functional.huber_loss(xyz, xyz_target)
        axangle_loss = 1 * torch.nn.functional.huber_loss(axangle, axangle_target)
        gripper_loss = 0.5 * torch.nn.functional.binary_cross_entropy(gripper, gripper_target)
        loss = {}
        loss['xyz_loss'] = xyz_loss
        loss['axangle_loss'] = axangle_loss
        loss['gripper_loss'] = gripper_loss
        loss['loss'] = xyz_loss + axangle_loss + gripper_loss
        return loss
    

def pose_to_T(pose):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T