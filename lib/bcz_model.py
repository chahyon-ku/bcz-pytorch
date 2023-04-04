import torch


class BCZModel(torch.nn.Module):
    def __init__(self, vision_network) -> None:
        super().__init__()
        self.vision_network = vision_network

    def forward(self, image, task_embed):
        image_embed = self.vision_network(image, task_embed)