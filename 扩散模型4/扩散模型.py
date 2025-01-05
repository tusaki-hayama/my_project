import torch
from torchvision import transforms
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.d_linear3 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        self.d_linear2 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
        )
        self.d_linear1 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU()
        )
        pass

    def forward(self, batch_image):
        x1 = self.linear1(batch_image)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        dx1 = self.d_linear3(x3)
        dx2 = self.d_linear2(dx1)
        dx3 = self.d_linear1(dx2)
        return dx3
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
# p = test_model.encoder_image(v_img)
