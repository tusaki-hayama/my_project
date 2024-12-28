import torch
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mse_loss = nn.MSELoss(reduction='sum')


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encode64_32 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.encode32_16 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.encode16_8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.encode8_4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decode4_8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decode8_16 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decode16_32 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decode32_64 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.pool32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool16 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg32 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg16 = nn.AvgPool2d(kernel_size=2, stride=2)
        pass

    def forward(self, batch_image):
        img32 = self.encode64_32(batch_image)
        img16 = self.encode32_16(img32)
        img8 = self.encode16_8(img16)
        img4 = self.encode8_4(img8)
        # print(img4.shape)
        d_img4 = self.decode4_8(img4)
        d_img8 = self.decode8_16(d_img4)
        d_img16 = self.decode16_32(d_img8)
        d_img32 = self.decode32_64(d_img16)

        down32 = self.pool32(batch_image)
        down16 = self.pool16(down32)
        ddown32 = self.pool32(d_img32)
        ddown16 = self.pool32(ddown32)

        avg32 = self.avg32(batch_image)
        avg16 = self.avg16(avg32)
        d_avg32 = self.avg32(d_img32)
        d_avg16 = self.avg16(d_avg32)

        loss64 = mse_loss(batch_image, d_img32)

        d_loss32 = mse_loss(down32, ddown32)
        d_loss16 = mse_loss(down16, ddown16)

        a_loss32 = mse_loss(avg32, d_avg32)
        a_loss16 = mse_loss(avg16, d_avg16)

        loss = loss64 + d_loss32 + d_loss16 + a_loss32 + a_loss16
        return d_img32, loss

    def encoder_image(self, batch_image):
        img32 = self.encode64_32(batch_image)
        img16 = self.encode32_16(img32)
        img8 = self.encode16_8(img16)
        img4 = self.encode8_4(img8)
        return img4.view((-1, 256 * 4 * 4))

    def decode_image(self, batch_v):
        img4 = batch_v.view((-1, 256, 4, 4))
        d_img4 = self.decode4_8(img4)
        d_img8 = self.decode8_16(d_img4)
        d_img16 = self.decode16_32(d_img8)
        d_img32 = self.decode32_64(d_img16)
        return d_img32

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
# p = test_model.encoder_image(v_img)
# print(p.shape)
# q = test_model.decode_image(p)
# print(q.shape)