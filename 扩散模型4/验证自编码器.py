import os

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img
from 自编码器模型 import auto_encoder
from torch import optim
from tqdm import tqdm
from PIL import Image

f_test = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
          r'\课程项目\archive\TEST')
test_data = load_data(f_test, 3000)
batch_size = 64
data2test = shuffle_and_div_batch(test_data, batch_size)
model = auto_encoder()
model.load_state_dict(torch.load('自编码器模型/checkpoint_auto_encoder.pt'))
model.eval()

predict_data = model.forward(data2test[0])

blackboard = Image.new('RGB', (64 * 2, 64 * batch_size))
for i in range(batch_size):
    blackboard.paste(tensor2img(data2test[0][i]), (64 * 0, 64 * i))
for i in range(batch_size):
    blackboard.paste(tensor2img(predict_data[i]), (64 * 1, 64 * i))
blackboard.show()