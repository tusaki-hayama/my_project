import os
import random

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, random_noise
from 模型 import auto_encoder
from torch import optim
from tqdm import tqdm

epoch = 0
epochs = 10000000000
lr = 1e-4
batch_size = 256
checkpoint_epoch = 0
checkpoint_epoch = None
checkpoint_loss = 203
checkpoint_loss = None
checkpoint_model = '模型日志/checkpoint_auto_encoder.pt'
checkpoint_model = None
best_loss = checkpoint_loss if checkpoint_loss is not None else float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_data(f_train)
val_data = load_data(f_val)

model = auto_encoder()
if checkpoint_model is not None:
    model.load_state_dict(torch.load(checkpoint_model))
    epoch = checkpoint_epoch
optimizer = optim.Adam(model.parameters(), lr=lr)
model.to(device)
mse_loss = nn.MSELoss(reduction='sum')
model_list = []

while epoch < epochs:
    epoch += 1
    model.train()
    train_loss = 0
    l_mse_loss = 0
    l1_loss = 0
    l2_loss = 0
    l3_loss = 0
    l4_loss = 0
    l5_loss = 0
    l6_loss = 0
    l7_loss = 0
    l8_loss = 0
    bs = random.randint(64, batch_size)
    data2train = shuffle_and_div_batch(train_data, bs)
    for gs in tqdm(range(data2train.shape[0])):
        X = data2train[gs] * random_noise(bs)
        Y = data2train[gs]
        optimizer.zero_grad()
        predict_Y, l_mse, l1, l2, l3, l4, l5, l6, l7, l8 = model.forward(X.to(device), Y.to(device))
        loss = random.random()*l_mse + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8
        train_loss += loss.item()
        l_mse_loss += l_mse.item()
        l1_loss += l1.item()  # 锐化损失
        l2_loss += l2.item()  # 平均损失
        l3_loss += l3.item()  # 左边缘损失
        l4_loss += l4.item()  # 右边缘损失
        l5_loss += l5.item()  # 水平损失
        l6_loss += l6.item()  # 竖直损失
        l7_loss += l7.item()  # 45度损失
        l8_loss += l8.item()  # 45度损失
        loss.backward()
        optimizer.step()
    print('第{}轮训练,单图片总卷积损失为:{}'
          .format(epoch, train_loss / train_data.shape[0]))
    print('第{}轮训练,单图片mse损失为:{}'
          .format(epoch, l_mse_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l1损失为:{}'
          .format(epoch, l1_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l2损失为:{}'
          .format(epoch, l2_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l3损失为:{}'
          .format(epoch, l3_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l4损失为:{}'
          .format(epoch, l4_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l5损失为:{}'
          .format(epoch, l5_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l6损失为:{}'
          .format(epoch, l6_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l7损失为:{}'
          .format(epoch, l7_loss / train_data.shape[0]))
    print('第{}轮训练,单图片l8损失为:{}'
          .format(epoch, l8_loss / train_data.shape[0]))

    model.eval()
    if epoch % 3 != 0:
        continue
    val_loss = 0
    data2val = shuffle_and_div_batch(val_data, batch_size)
    for gs in tqdm(range(data2val.shape[0])):
        X = data2val[gs] * random_noise(batch_size)
        Y = data2val[gs]
        predict_Y, l_mse, l1, l2, l3, l4, l5, l6, l7, l8 = model.forward(X.to(device), Y.to(device))
        val_loss += l_mse.item()
    print('第{}轮验证,单图片mse损失为:{}'
          .format(epoch, val_loss / val_data.shape[0]))
    if val_loss / val_data.shape[0] < best_loss:
        best_loss = val_loss / val_data.shape[0]
        print('正在保存模型参数')
        torch.save(model.state_dict(), '模型日志/checkpoint_auto_encoder.pt')
        model_name = '模型日志/model_epoch_{}_loss{}.pt'.format(epoch, val_loss / val_data.shape[0])
        torch.save(model.state_dict(), model_name)
        model_list.append(model_name)
        print('模型参数保存完成')
        if len(model_list) > 10:
            try:
                del_name = model_list[0]
                model_list.pop(0)
                os.remove((del_name))
            except Exception as e:
                print(e.args)
