import torch
from torch import nn
from torchvision import transforms
from 工具类 import load_data

max_step = 10
batch_size = 64
epochs = 1000000000
l = torch.linspace(0, 1, steps=max_step)
print('步幅:{}'.format(l))

f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')

train_data = load_data(f_train, batch_size, 3, 64, 64,
                       '训练集加载完成')





