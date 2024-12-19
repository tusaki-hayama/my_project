import torch
from torch import nn
from torchvision import transforms


class diffusion_model(nn.Module):
    def __init__(self):
        super(diffusion_model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,6,3),
            nn.MaxPool2d(6),
            nn.ReLU(),

        )
        pass

    def forward(self, batch_img):
        x = self.conv(batch_img)

        print(x.shape)
        return
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
test_model.forward(v_img)
