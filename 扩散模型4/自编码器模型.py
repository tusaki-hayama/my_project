import torch
from torchvision import transforms
from torch import nn


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 768),
            nn.Unflatten(dim=1, unflattened_size=(48, 4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 48, 2, 2),
            nn.ConvTranspose2d(48, 24, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 24, 2, 2),
            nn.ConvTranspose2d(24, 12, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 12, 2, 2),
            nn.ConvTranspose2d(12, 6, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 6, 2, 2),
            nn.ConvTranspose2d(6, 3, 3, padding=1),
            nn.ReLU(),
        )
        pass

    def forward(self, batch_image):
        encode_v = self.encoder(batch_image)
        build_image = self.decoder(encode_v)
        return build_image

    def encoder_image(self, batch_image):
        encode_v = self.encoder(batch_image)
        return encode_v

    def decode_image(self, batch_v):
        build_image = self.decoder(batch_v)
        return build_image

# from PIL import Image
#
# img2tensor = transforms.ToTensor()
# img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
# v_img = img2tensor(img)
# print(v_img.shape)
# v_img = v_img.view(-1, 3, 64, 64)
# print(v_img.shape)
# test_model = auto_encoder()
# b_steps = torch.zeros((1, 2))
# b_steps[:, 0] = 0.1
# b_steps[:, 1] = 0.2
# print('模型内参数:')
# test_model.forward(v_img)
