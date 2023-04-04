import os
import pickle
import torch
import torch.utils.data
import rlbench.utils
import hydra
import typing
import rlbench.action_modes.arm_action_modes
import numpy as np
import transforms3d as t3d
from PIL import Image
import torchvision.transforms.functional


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, variations, demos_config, action_mode, task_embeds):
        self.data = None
        self.action_mode = action_mode
        self.demos_config = demos_config
        self.variations = variations
        self.len = sum([len(rlbench.utils.get_stored_demos(**demos_config, variation_number=variation)) for variation in variations])

        self.task_embeds = task_embeds

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            self.data = self._load_data()

        return self.data[index]
    
    def _load_data(self) -> typing.List[typing.Tuple[torch.Tensor, torch.Tensor]]:
        # print(self.action_mode_config)
        # action_mode: rlbench.ActionMode = hydra.utils.instantiate(self.action_mode_config)
        data = []
        for variation in self.variations:
            demos = rlbench.utils.get_stored_demos(variation_number=variation, **self.demos_config)
            for demo in demos:
                for i_obs in range(len(demo)):
                    image = torchvision.transforms.functional.to_tensor(Image.open(demo[i_obs].front_rgb))

                    if isinstance(self.action_mode.arm_action_mode, rlbench.action_modes.arm_action_modes.EndEffectorPoseViaIK):
                        arm_curr = demo[i_obs].gripper_pose
                        arm_next = demo[min(i_obs + 1, len(demo) - 1)].gripper_pose
                        if self.action_mode.arm_action_mode._absolute_mode:
                            arm_action = arm_next
                        else:
                            world_T_curr = t3d.affines.compose(arm_curr[:3], t3d.quaternions.quat2mat(arm_curr[3:]), [1, 1, 1])
                            world_T_next = t3d.affines.compose(arm_next[:3], t3d.quaternions.quat2mat(arm_next[3:]), [1, 1, 1])
                            curr_T_next = np.linalg.inv(world_T_curr) @ world_T_next
                            arm_action = np.concatenate([t3d.affines.decompose(curr_T_next)[0], t3d.quaternions.mat2quat(t3d.affines.decompose(curr_T_next)[1])])
                    else:
                        arm_curr = demo[i_obs].joint_positions
                        arm_next = demo[min(i_obs + 1, len(demo) - 1)].joint_positions
                        if self.action_mode.arm_action_mode._absolute_mode:
                            arm_action = arm_next
                        else:
                            arm_action = arm_next - arm_curr
                    action = np.concatenate([arm_action, torch.tensor([demo[min(i_obs + 1, len(demo) - 1)].gripper_open])])
                    data.append((image, self.task_embeds['reach_target'][variation], action))
        return data