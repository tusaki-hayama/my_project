import os

import torch
from torchvision import transforms
from torch import nn, optim
from 工具类 import load_tensor_data, random_noise
from 扩散模型 import diffusion_model
from tqdm import tqdm

batch_size = 64
study_rare = 1e-5
epochs = 100000000000
epoch = 0
best_val_loss = float('inf')
f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_tensor_data(f_train, batch_size, 3, 64, 64,
                              '训练集加载完成')
val_data = load_tensor_data(f_val, batch_size, 3, 64, 64,
                            '测试集加载完成')

model = diffusion_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_data.to(device)
val_data.to(device)
optimizer = optim.Adam(model.parameters(), lr=study_rare)
mse_loss = nn.MSELoss(reduction='sum')
model_list = []
while epoch < epochs:
    model.train()
    train_loss = 0
    for bs in tqdm(range(train_data.shape[0])):
        optimizer.zero_grad()
        noise = random_noise(batch_size)
        Y = train_data[bs].to(device)
        X = train_data[bs].to(device) * noise.to(device)
        pY = model.forward(X)
        loss = mse_loss(pY, Y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('本{}轮训练单图片损失为:{}'.format(epoch, train_loss / (train_data.shape[0] * train_data.shape[1])))
    pass
    if epoch % 3 != 0:
        continue
    model.eval()
    val_loss = 0
    for bs in tqdm(range(val_data.shape[0])):
        noise = random_noise(batch_size)
        Y = val_data[bs].to(device)
        X = val_data[bs].to(device) * noise.to(device)
        pY = model.forward(X)
        loss = mse_loss(pY, Y)
        val_loss += loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_name = 'best_model{}.pt'.format(val_loss)
        model_list.append(model_name)
        torch.save(model.state_dict(), model_name)
        if len(model_list) > 10:
            del_name = model_list[0]
            model_list.pop(0)
            os.remove(del_name)

    epoch += 1
