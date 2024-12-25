import torch
from torchvision import transforms
from torch import nn


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,6,3),
            nn.ReLU(),
        )
        pass

    def forward(self):
        pass
