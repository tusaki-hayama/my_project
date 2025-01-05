import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse_loss = nn.MSELoss(reduction='sum')


def const_conv_layer(kernel, channel=3):
    kernel = kernel.unsqueeze(0).repeat(channel, 1, 1)
    layer = nn.Conv2d(3, 1, 3, padding=1)
    layer.eval()
    layer.weight = nn.Parameter(kernel)
    return layer
    pass


#
# print(const_conv_layer(torch.Tensor([[1,1,1],[1,1,1],[1,1,1]])))

class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        pass
        self.k1 = const_conv_layer(torch.Tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]) / 3)
        self.k2 = const_conv_layer(torch.Tensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]) / 27)
        self.k3 = const_conv_layer(torch.Tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]))
        self.k4 = const_conv_layer(torch.Tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]))
        self.k5 = const_conv_layer(torch.Tensor([
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1],
        ]))
        self.k6 = const_conv_layer(torch.Tensor([
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1],
        ]))
        self.k7 = const_conv_layer(torch.Tensor([
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2],
        ]))
        self.k8 = const_conv_layer(torch.Tensor([
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1],
        ]))

    def forward(self, batch_x, batch_y):
        encode_x = self.encoder(batch_x)
        decode_x = self.decoder(encode_x)
        loss_mse = mse_loss(batch_y, decode_x)
        loss_k1 = mse_loss(self.k1(batch_y), self.k1(decode_x))
        loss_k2 = mse_loss(self.k2(batch_y), self.k2(decode_x))
        loss_k3 = mse_loss(self.k3(batch_y), self.k3(decode_x))
        loss_k4 = mse_loss(self.k4(batch_y), self.k4(decode_x))
        loss_k5 = mse_loss(self.k5(batch_y), self.k5(decode_x))
        loss_k6 = mse_loss(self.k6(batch_y), self.k6(decode_x))
        loss_k7 = mse_loss(self.k7(batch_y), self.k7(decode_x))
        loss_k8 = mse_loss(self.k8(batch_y), self.k8(decode_x))
        return encode_x, loss_mse, loss_k1, loss_k2, loss_k3, loss_k4, loss_k5, loss_k6, loss_k7, loss_k8


from PIL import Image

img2tensor = transforms.ToTensor()
img = Image.open(r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习\课程项目\archive\TRAIN\4.jpg')
v_img = img2tensor(img)
print(v_img.shape)
v_img = v_img.view(-1, 3, 64, 64)
print(v_img.shape)
test_model = auto_encoder()
print('模型内参数:')
# test_model.forward(v_img)
