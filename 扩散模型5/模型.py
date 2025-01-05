import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse_loss = nn.MSELoss(reduction='sum')
l_relu = 1e-10


#
# print(const_conv_layer(torch.Tensor([[1,1,1],[1,1,1],[1,1,1]])))

class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4,stride=2,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        pass
        self.conv_loss = self.const_conv_layer()

    def const_conv_layer(self):
        layer = nn.Conv2d(3, 3, 3, groups=3,padding=1)
        layer.weight.data = nn.Parameter(torch.randn(3, 1, 3, 3))
        # print(layer)
        return layer.eval().to(device)
        pass

    def forward(self, batch_x, batch_y):
        encode_x = self.encoder(batch_x)
        decode_x = self.decoder(encode_x)
        loss_mse = mse_loss(batch_y, decode_x)
        # print(batch_y.shape, decode_x.shape)
        loss_conv = mse_loss(self.conv_loss(batch_y), self.conv_loss(decode_x))
        return decode_x, loss_mse,loss_conv

    def change_loss(self):
        self.conv_loss = self.const_conv_layer()
        print('损失卷积已更换')
        pass


# from PIL import Image
#
# img2tensor = transforms.ToTensor()
# img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
# v_img = img2tensor(img)
# print(v_img.shape)
# v_img = v_img.view(-1, 3, 64, 64)
# print(v_img.shape)
# test_model = auto_encoder().to(device)
# print('模型内参数:')
# test_model.forward(v_img.to(device), v_img.to(device))
