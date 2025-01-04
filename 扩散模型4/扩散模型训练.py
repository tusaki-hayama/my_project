import os
import random
import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img, random_noise
from 自编码器模型 import auto_encoder
from 扩散模型 import diffusion_model
from torch import optim
from tqdm import tqdm
from PIL import Image

epochs = 10000000000
epoch = 0
batch_size = 64
time_step = 20
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auto_model = auto_encoder()
auto_model.to(device)
auto_model.load_state_dict(torch.load('自编码器模型/checkpoint_auto_encoder.pt'))
auto_model.eval()
diffusion_model = diffusion_model()
diffusion_model.to(device)
checkpoint_loss = 203
checkpoint_loss = None
checkpoint_model = '自编码器模型/checkpoint_auto_encoder.pt'
checkpoint_model = None
best_loss = checkpoint_loss if checkpoint_loss is not None else float('inf')
mse_loss = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_data(f_train)
val_data = load_data(f_val)
beta = torch.linspace(0, 1, time_step)
alpha = 1 - beta
alpha_mul = torch.cumprod(alpha, dim=0)
print(alpha_mul)
while epoch < epochs:
    epoch += 1
    # 准备训练数据
    data2train = shuffle_and_div_batch(train_data, random.randint(64, batch_size))
    noise_data2train = data2train * random_noise(batch_size)
    random_t = random.randint(0, time_step - 1)

    # 训练
    diffusion_model.train()
    train_loss = 0

    break

    pass
