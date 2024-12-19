import torch
from torch import nn, optim
from torchvision import transforms
from 工具类 import load_data, img2tensor, tensor2img, random_noise
from 扩散模型 import diffusion_model
from tqdm import tqdm
import random

model = diffusion_model()
model.load_state_dict(torch.load('first_model.pt'))

max_step = 10
batch_size = 64
study_rare = 1e-4
epochs = 1000000000
l = torch.linspace(0, 1, steps=max_step)

f_test = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
          r'\课程项目\archive\TEST')
test_data = load_data(f_test, batch_size, 3, 64, 64,
                      '测试集加载完成')

i = 0
r_noise = random_noise()
r_tensor = test_data[random.randint(0, 10)][random.randint(0, 10)]*r_noise
model.eval()
b_steps = torch.zeros((1, 2))
b_steps[:, 0] = 9
b_steps[:, 1] = 8
re_image = model.forward(r_tensor.view(1, 3, 64, 64), b_steps)
tensor2img(r_tensor.view(3, 64, 64)).show()
tensor2img(re_image.view(3, 64, 64)).show()
