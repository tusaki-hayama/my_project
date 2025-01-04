import os

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img
from 自编码器模型 import auto_encoder
from 扩散模型 import diffusion_model
from torch import optim
from tqdm import tqdm
from PIL import Image

epochs = 10000000000
epoch = 0
batch_size = 64
step = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auto_model = auto_encoder()
auto_model.to(device)
auto_model.load_state_dict(torch.load('自编码器模型/checkpoint_auto_encoder.pt'))
auto_model.eval()
f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_data(f_train)
val_data = load_data(f_val)









