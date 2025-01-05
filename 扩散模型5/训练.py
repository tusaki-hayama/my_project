import os
import random

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch
from 模型 import auto_encoder
from torch import optim
from tqdm import tqdm

epoch = 0
epochs = 10000000
lr = 1e-4
batch_size = 256
checkpoint_loss = 203
checkpoint_loss = None
checkpoint_model = '自编码器模型/checkpoint_auto_encoder.pt'
checkpoint_model = None
best_loss = checkpoint_loss if checkpoint_loss is not None else float('inf')




