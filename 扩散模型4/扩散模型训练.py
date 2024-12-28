import os

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img
from 自编码器模型 import auto_encoder
from torch import optim
from tqdm import tqdm
from PIL import Image

epochs = 10000000000
epoch = 0
batch_size = 64
step = 1000
model = auto_encoder()
model.load_state_dict(torch.load('自编码器模型/checkpoint_auto_encoder.pt'))
model.eval()










