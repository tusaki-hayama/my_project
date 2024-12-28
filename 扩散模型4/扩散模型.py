import torch
from torchvision import transforms
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model,self).__init__()
        pass
    def forward(self,batch_image):
        pass






