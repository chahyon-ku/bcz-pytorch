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
        self.fc1 = nn.Linear(1025, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 30)

        # gripper regression
        self.fcg = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, curr_gripper):
        # pass x through resnet
        x = self.resnet(x)
        # pass y through resnet2
        y = self.resnet2(y)
        # flatten
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        # concatenate
        avg = torch.cat((x, y, curr_gripper), 1)

        # pass through xyz regression layers
        x = self.fc1(avg)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.reshape(x, (-1, 10, 3))

        # pass through gripper classifier (open or closed)
        g = self.fc1(avg)
        g = self.relu(g)
        g = self.fc2(g)
        g = self.relu(g)
        g = self.fcg(g)
        g = self.sigmoid(g)

        return x, g
    
    def get_action(self, obs, task_embed):
        # obs: rlbench.observation.Observation
        # task_embed: (512,)
        device = 'cuda:0'
        image = TF.to_tensor(Image.fromarray(obs.front_rgb)).unsqueeze(0).to(device)
        image2 = TF.to_tensor(Image.fromarray(obs.left_shoulder_rgb)).unsqueeze(0).to(device)
        # image2 = TF.to_tensor(Image.fromarray(obs.wrist_rgb)).unsqueeze(0).to(device)
        curr_gripper = torch.tensor([obs.gripper_open]).unsqueeze(0).to(device)
        xyz, gripper = self.forward(image, image2, curr_gripper)
        
        # show this image
        # image = image.detach().cpu().numpy()[0]
        # image = np.transpose(image, (1, 2, 0))
        # plt.imshow(image)
        # plt.show()

        # xyz: (batch_size, 10, 3)
        # axangle: (batch_size, 10, 4)
        # print('z', xyz[0, :, 2].detach().cpu().numpy().round(2))
        xyz = xyz.detach().cpu().numpy()[0, 0]
        # axangle = axangle.detach().cpu().numpy()[0, 0]
        print('gripper', torch.round(gripper).detach().cpu().numpy())
        gripper = gripper.detach().cpu().numpy()[0, 0]

        curr_xyz = obs.gripper_pose[:3]
        curr_quat = obs.gripper_pose[3:]
        # curr_axangle = R.from_quat(curr_quat).as_rotvec()

        curr_xyz += xyz/40
        # axangle = curr_axangle + axangle
        # axangle = curr_axangle
        # quat = R.from_rotvec(axangle).as_quat()
        
        action = np.concatenate([curr_xyz, curr_quat, [gripper]])

        return action
    
    def loss(self, input, target):
        xyz, gripper = input
        xyz_target, gripper_target = target
        xyz_loss = 10 * torch.nn.functional.huber_loss(xyz, xyz_target)
        gripper_loss = 5 * torch.nn.functional.binary_cross_entropy(gripper, gripper_target)
        # gripper_loss = 5 * torch.nn.functional.binary_cross_entropy_with_logits(gripper, gripper_target)
        loss = {}
        loss['xyz_loss'] = xyz_loss
        loss['gripper_loss'] = gripper_loss
        loss['total_loss'] = xyz_loss + gripper_loss
        return loss

if __name__ == '__main__':
    # create model
    model = BC()
    # test model
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    z, g = model(x, y)
    print(z, g)
    # get action
    # action = model.get_action(x, y)