import torch
from torchvision import transforms
from torch import nn


class Coder(nn.Module):
    def __init__(self):
        super(Coder, self).__init__()
        self.layer_catch_noise = nn.Sequential(
            nn.Linear(28 * 28 * 3, 176),
            nn.ReLU(),
            nn.Linear(176, 31)
        )
        self.layer_catch_feature = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 12, kernel_size=3),
        )
        self.layer_tips = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.d_layer_noise = nn.Sequential(
            nn.Linear(2071, 31),
            nn.Linear(31, 176),
            nn.ReLU(),
            nn.Linear(176, 28 * 28 * 3)
        )
        self.d_layer_feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2071, 11 * 11 * 12),
            nn.Unflatten(dim=1, unflattened_size=(12, 11, 11)),
            nn.ConvTranspose2d(12, 6, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2),
            nn.ConvTranspose2d(6, 3, kernel_size=3)
        )

    def forward(self, input_BCHW):
        noise = self.layer_catch_noise(input_BCHW.view(-1, 1, 28 * 28 * 3))
        feature = self.layer_catch_feature(input_BCHW)
        tips = self.layer_tips(input_BCHW)
        zip_v = torch.cat(
            (noise.view(-1, 1, 31),
             feature.view(-1, 1, 11 * 11 * 12),
             tips.view(-1, 1, 14 * 14 * 3))
            , dim=2)
        # print(zip_v.shape)
        feature_tensor = self.d_layer_feature(zip_v)
        noise_tensor = self.d_layer_noise(zip_v)
        # print(feature_tensor.shape)
        # print(noise_tensor.shape)
        result = torch.sigmoid(feature_tensor + noise_tensor.view(-1, 3, 28, 28))
        return result
        pass

    pass


# c = Coder()
# from PIL import Image
#
# pimg = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\train_images\0.jpg')
# vimg = transforms.ToTensor()(pimg)
# print(vimg)
# print(vimg)
# vimg = vimg.view(1,3,28,28)
# c.forward(vimg)
