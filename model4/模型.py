from torchvision import transforms
from torch import nn
import torch
from model4.配置 import args

weight = torch.tensor([0.3, 0.6, 0.1]).view(1, 3, 1, 1)


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.layer_catch_noise = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.layer_catch_feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer_tips = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.d_layer_noise = nn.Sequential(
            nn.Linear(5900, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64 * 3)
        )
        self.d_layer_feature = nn.Sequential(
            nn.Linear(5900, 2700),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(12, 15, 15)),
            nn.ConvTranspose2d(12, 6, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 6, 2, 2),
            nn.Conv2d(6, 6, 3),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 2, 2)
        )

    pass

    def forward(self, batch_img):
        gray_img = torch.sum(batch_img * weight, dim=1, keepdim=True)
        noise = self.layer_catch_noise(gray_img)
        feature = torch.flatten(self.layer_catch_feature(gray_img), start_dim=1)
        tips = torch.flatten(self.layer_tips(batch_img), start_dim=1)
        # print(noise.shape)
        # print(feature.shape)
        # print(tips.shape)
        combine_v = torch.cat([noise, feature, tips], dim=1)
        # print(combine_v.shape)
        d_noise = self.d_layer_noise(combine_v).view((-1, 3, 64, 64))
        d_feature = self.d_layer_feature(combine_v)
        # print(d_feature.shape)
        # print(d_noise.shape)
        return torch.sigmoid(d_noise+d_feature)
        pass


# from PIL import Image
#
# img2tensor = transforms.ToTensor()
# img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
# v_img = img2tensor(img)
# print(v_img.shape)
# v_img = v_img.view(-1, 3, 64, 64)
# print(v_img.shape)
# test_model = auto_encoder()
# test_model.forward(v_img)
