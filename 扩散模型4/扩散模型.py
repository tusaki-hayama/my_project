import torch
from torchvision import transforms
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )


        pass
    def forward(self,batch_image):
        pass






