import random

import torch
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
        self.gen_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, 2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(12, 36, 3, 2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(36, 48, 2),
        )
        self.gen_decoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, 2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(12, 36, 3, 2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(36, 48, 2),
        )
        self.encoder = nn.Conv2d(3, 16, 3, padding=1)
        self.decoder = nn.Conv2d(16, 3, 3, padding=1)

        # self.gen_decoder = nn.Sequential(
        #
        # )
        # self.decoder = nn.Sequential(
        #
        # )
        pass

    def forward(self, batch_image):  # (bs,3,64,64)
        e = self.gen_encoder(batch_image)
        e = torch.sum(e, dim=0).view((16, 3, 3, 3)).clone()
        e = (e  - torch.min(e)) / (torch.max(e)-torch.min(e))
        # print(e.shape)
        d = self.gen_decoder(batch_image)
#         print(d.shape)
        d = torch.sum(d, dim=0).view((3, 16, 3, 3)).clone()
        d = (d - torch.min(d)) / (torch.max(d) - torch.min(d))
        self.encoder.weight = nn.Parameter(e)
        self.decoder.weight = nn.Parameter(d)
        p_img = self.encoder(batch_image)
        p_img = torch.relu(p_img)
        p_img = self.decoder(p_img)
        p_img = (p_img - torch.min(p_img)) / (torch.max(p_img) - torch.min(p_img))
        return p_img
        pass


from PIL import Image

img2tensor = transforms.ToTensor()
img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
v_img = img2tensor(img)
print(v_img.shape)
v_img = v_img.view(-1, 3, 64, 64)
print(v_img.shape)
test_model = diffusion_model()
b_steps = torch.zeros((1, 2))
b_steps[:, 0] = 0.1
b_steps[:, 1] = 0.2
print('模型内参数:')
test_model.forward(v_img)
