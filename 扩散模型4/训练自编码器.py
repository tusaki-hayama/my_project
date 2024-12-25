import os

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch
from 自编码器模型 import auto_encoder
from torch import optim
from tqdm import tqdm

epoch = 0
epochs = 10000000
lr = 1e-5
batch_size = 128
checkpoint_loss = 230
checkpoint_model = '自编码器模型/checkpoint_auto_encoder.pt'
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
optimizer = optim.Adam(model.parameters(), lr=lr)
model.to(device)
mse_loss = nn.MSELoss(reduction='sum')
model_list = []

while epoch < epochs:
    model.train()
    train_loss = 0
    data2train = shuffle_and_div_batch(train_data, batch_size)
    for gs in tqdm(range(data2train.shape[0])):
        X = data2train[gs].to(device)
        optimizer.zero_grad()
        Y = model.forward(X)
        loss = mse_loss(Y, X)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('第{}轮训练,单图片损失为:{}'.format(epoch, train_loss / train_data.shape[0]))
    if epoch % 3 != 0:
        epoch += 1
        continue

    model.eval()
    val_loss = 0
    data2val = shuffle_and_div_batch(val_data, batch_size)
    for gs in tqdm(range(data2val.shape[0])):
        X = data2val[gs].to(device)
        Y = model.forward(X)
        loss = mse_loss(Y, X)
        val_loss += loss.item()
    print('第{}轮验证,单图片损失为:{}'.format(epoch, val_loss / val_data.shape[0]))
    if val_loss / val_data.shape[0] < best_loss:
        best_loss = val_loss / val_data.shape[0]
        print('正在保存模型参数')
        torch.save(model.state_dict(), '自编码器模型/checkpoint_auto_encoder.pt')
        model_name = '自编码器模型/model_epoch_{}_loss{}.pt'.format(epoch, val_loss / val_data.shape[0])
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

    epoch += 1
