import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt


class BCZModel(torch.nn.Module):
    def __init__(self, vision_encoders, policy_decoder, views, task_embed_std, fusion, action_scale, model_path=None) -> None:
        super().__init__()
        self.vision_encoders = vision_encoders
        self.policy_decoder = policy_decoder
        self.views = views
        self.task_embed_std = task_embed_std
        self.fusion = fusion
        self.action_scale = action_scale
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def forward(self, images, task_embed):
        # view: [B, O, C, H, W]
        task_embed += torch.randn_like(task_embed) * self.task_embed_std
        processed_images = {}
        for view_mode, frames in images.items():
            view = view_mode.split('_')[0]
            if view not in processed_images:
                processed_images[view] = []
            processed_images[view].append(frames)
        # modes: V, [B, O, C, H, W]
        if self.fusion == 'early':
            images = [torch.concat(modes, dim=2) for view, modes in processed_images.items()]
            # images: V', [B, O, C', H, W]
        elif self.fusion == 'late':
            images = [mode for modes in processed_images.values() for mode in modes]
            # images: V, [B, O, C, H, W]
        else:
            raise NotImplementedError
        
        for i in range(len(images)):
            if i == 0:
                image_embed = self.vision_encoders[i](images[i], task_embed)
            else:
                image_embed += self.vision_encoders[i](images[i], task_embed)
        # image_embed: V, [B, O, V, D]
        # image_embed = torch.sum(image_embed)
        # print(image_embed.shape)
        # image_embed: [B, O, 1, D]
        xyz, axangle, gripper = self.policy_decoder(image_embed)
        # print(xyz, axangle, gripper)
        return xyz, axangle, gripper

    def get_action(self, obs, task_embed):
        # obs: rlbench.observation.Observation
        # task_embed: (768,)
        # images = torch.stack([TF.to_tensor(Image.fromarray(obs.__dict__[view])) for view in self.views], dim=0)[None, None, ...].to(task_embed)
        images = {view: ([obs.__dict__[view].transpose(2, 0, 1).astype(np.float32) / 255]
                         if 'rgb' in view else
                         [obs.__dict__[view][None, ...]])
                  for view in self.views}
        # images = {view: ([rlbench.backend.utils.image_to_float_array(frame)[None, ...].astype(np.float32) for frame in frames]
        #                  if 'depth' in view else
        #                  [np.array(frame.convert('RGB')).transpose(2, 0, 1).astype(np.float32) / 255.0 for frame in frames])
        #                  for view, frames in images.items()}
        # for view, frames in images.items():
        #     print(frames[0].shape, frames[0].max(), frames[0].min())
        #     plt.imshow(frames[0].transpose(1, 2, 0))
        #     plt.show()
        images = {view: np.stack(frames, axis=0)[None, ...] for view, frames in images.items()}
        images = {view: torch.from_numpy(frames).to(task_embed) for view, frames in images.items()}

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
        xyz = curr_xyz + xyz * self.action_scale[0]
        axangle = curr_axangle + axangle * self.action_scale[1]
        # axangle = curr_axangle
        quat = R.from_rotvec(axangle).as_quat()
        
        action = np.concatenate([xyz, quat, gripper])

        return action

    def loss(self, input, target, weights):
        xyz, axangle, gripper = input
        xyz_target, axangle_target, gripper_target = target
        xyz_loss = weights[0] * torch.nn.functional.huber_loss(xyz, xyz_target)
        axangle_loss = weights[1] * torch.nn.functional.huber_loss(axangle, axangle_target)
        gripper_loss = weights[2] * torch.nn.functional.binary_cross_entropy(gripper, gripper_target)
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