import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

class BC(nn.Module):
    def __init__(self, freeze = False):
        super().__init__()

        # get pretrained resnet18 model
        self.resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        self.resnet2 = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        # get new model that stops at avgpool layer
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet2 = torch.nn.Sequential(*list(self.resnet2.children())[:-1])
        # freeze resnet layers
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.resnet2.parameters():
                param.requires_grad = False

        # create new layers
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x, y):
        # pass x through resnet
        x = self.resnet(x)
        # pass y through resnet2
        y = self.resnet2(y)
        # flatten
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        # concatenate
        x = torch.cat((x, y), 1)
        # pass through new layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def get_action(self, obs, task_embed):
        # obs: rlbench.observation.Observation
        # task_embed: (512,)
        device = 'cuda:1'
        # image = TF.to_tensor(Image.fromarray(obs.front_rgb)).unsqueeze(0).to(device)
        # image2 = TF.to_tensor(Image.fromarray(obs.left_shoulder_rgb)).unsqueeze(0).to(device)
        image = TF.to_tensor(Image.fromarray(obs.left_shoulder_rgb)).unsqueeze(0).to(device)
        image2 = TF.to_tensor(Image.fromarray(obs.right_shoulder_rgb)).unsqueeze(0).to(device)
        xyz = self.forward(image, image2)
        
        # show this image
        # image = image.detach().cpu().numpy()[0]
        # image = np.transpose(image, (1, 2, 0))
        # plt.imshow(image)
        # plt.show()

        # xyz: (batch_size, 10, 3)
        # axangle: (batch_size, 10, 4)
        # gripper: (batch_size, 10, 1)
        xyz = xyz.detach().cpu().numpy()[0]
        # axangle = axangle.detach().cpu().numpy()[0, 0]
        # gripper = gripper.detach().cpu().numpy()[0, 0]
        curr_xyz = obs.gripper_pose[:3]
        curr_quat = obs.gripper_pose[3:]
        # curr_axangle = R.from_quat(curr_quat).as_rotvec()
        # xyz = curr_xyz + xyz
        # print('yz prediction: ', yz)
        # print('yz shape', yz.shape)
        curr_xyz += xyz/40
        # axangle = curr_axangle + axangle
        # axangle = curr_axangle
        # quat = R.from_rotvec(axangle).as_quat()
        
        action = np.concatenate([curr_xyz, curr_quat, [1]])

        return action

if __name__ == '__main__':
    # create model
    model = BC()
    # test model
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    z = model(x, y)
    print(z)