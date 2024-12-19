import torch
from torch import nn
from torchvision import transforms


class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 9, 3, stride=3),
            nn.ReLU(),
            nn.Conv2d(9, 27, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(27, 64, 2, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.line = nn.Sequential(
            nn.Linear(1026, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 27, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(27, 9, 3, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, 4, 2),
            nn.Sigmoid()
        )
        pass

    def forward(self, batch_img, batch_steps):
        x = self.conv(batch_img)
        x = torch.cat([x, batch_steps], dim=1)
        x = self.line(x)
        x = x.view((-1, 64, 4, 4))
        x = self.de_conv(x)
        return x
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
test_model.forward(v_img, b_steps)
