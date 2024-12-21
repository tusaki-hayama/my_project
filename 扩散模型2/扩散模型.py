import torch
from torchvision import transforms
from torch import nn


class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 12, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 24, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.d_conv4 = nn.Sequential(
            nn.ConvTranspose2d(48, 48, 2, 2),
            nn.ConvTranspose2d(48, 24, 3, padding=1),
            nn.ReLU()
        )
        self.d_conv3 = nn.Sequential(
            nn.ConvTranspose2d(24, 24, 2, 2),
            nn.ConvTranspose2d(24, 12, 3, padding=1),
            nn.ReLU()
        )
        self.d_conv2 = nn.Sequential(
            nn.ConvTranspose2d(12, 12, 2, 2),
            nn.ConvTranspose2d(12, 6, 3, padding=1),
            nn.ReLU()
        )
        self.d_conv1 = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 2, 2),
            nn.ConvTranspose2d(6, 3, 3, padding=1),
            nn.ReLU()
        )
        pass

    def forward(self, batch_image):  # (bs,3,64,64)
        x1 = self.conv1(batch_image)
        # print(x1.shape)
        x2 = self.conv2(x1)
#         print(x2.shape)
        x3 = self.conv3(x2)
#         print(x3.shape)
        x4 = self.conv4(x3)
#         print(x4.shape)
#         print("back:")
        dx4 = self.d_conv4(x4)
#         print(dx4.shape)
        dx3 = self.d_conv3(dx4 + x3)
#         print(dx3.shape)
        dx2 = self.d_conv2(dx3 + x2)
#         print(dx2.shape)
        dx1 = self.d_conv1(dx2 + x1)
        # print(dx1.shape)
        return dx1
        pass


# from PIL import Image
#
# img2tensor = transforms.ToTensor()
# img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
# v_img = img2tensor(img)
# print(v_img.shape)
# v_img = v_img.view(-1, 3, 64, 64)
# print(v_img.shape)
# test_model = diffusion_model()
# b_steps = torch.zeros((1, 2))
# b_steps[:, 0] = 0.1
# b_steps[:, 1] = 0.2
# print('模型内参数:')
# test_model.forward(v_img)
