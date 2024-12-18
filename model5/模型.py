from torchvision import transforms
from torch import nn
import torch
from model4.配置 import args

weight = (torch.tensor([0.3, 0.6, 0.1]).view(1, 3, 1, 1)).to(args.device)


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 128, kernel_size=1),
            nn.ReLU()
        )
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.ReLU()
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.ReLU()
        )
        self.d_conv3 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.ReLU()
        )

    pass

    def forward(self, batch_img):
        b_img = batch_img
        x = self.conv1(b_img.permute(0, 2, 1, 3))
        x = self.conv2(x.permute(0, 3, 2, 1))
        x = self.conv3(x.permute(0, 2, 1, 3))
        x = self.d_conv1(x.permute(0, 2, 1, 3))
        x = self.d_conv2(x.permute(0, 3, 2, 1))
        x = self.d_conv3(x.permute(0, 2, 1, 3))
        return torch.sigmoid(x)
        pass


from PIL import Image

img2tensor = transforms.ToTensor()
img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
v_img = img2tensor(img)
print(v_img.shape)
v_img = v_img.view(-1, 3, 64, 64)
print(v_img.shape)
test_model = auto_encoder()
test_model.forward(v_img)
