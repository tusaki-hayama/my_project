import os

import numpy
from PIL import Image
from 自编码器 import self_encoder
import torch
import torch.nn as nn
import torch.nn.functional as func
import tqdm
from torch import optim
from torchvision import transforms
from 图像信息 import train_data_folder, test_data_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
mes_loss = nn.CrossEntropyLoss(reduction='sum')
model_name = '自编码器.pt'
epoch = 10000000
batch_size = 16
lr = 1e-5

model = self_encoder()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
slr = torch.optim.lr_scheduler
img2tensor = transforms.ToTensor()
train_names = os.listdir(train_data_folder)[:]
train_num = len(train_names)
test_names = os.listdir(test_data_folder)[:]
test_num = len(test_names)
best_loss = float('inf')
"""
加载训练集
"""
print('加载训练集')
train_set = torch.zeros((train_num, 3, 28, 28))
for i in tqdm.tqdm(range(train_num)):
    img = Image.open(train_data_folder + '\\' + train_names[i])
    img_v = img2tensor(img)
    train_set[i] = img_v
train_set = train_set.view(train_num, 1, -1).to(device)
"""
加载测试集
"""
print('加载测试集')
test_set = torch.zeros((test_num, 3, 28, 28))
for i in tqdm.tqdm(range(test_num)):
    img = Image.open(test_data_folder + '\\' + test_names[i])
    img_v = img2tensor(img)
    test_set[i] = img_v
test_set = test_set.view(test_num, 1, -1).to(device)


for r in range(epoch):
    model.train()
    train_loss = 0
    for i in tqdm.tqdm(range(train_num)):
        optimizer.zero_grad()
        img = train_set[i].clone()
        # train_img = Image.open(train_data_folder + '\\' + train_names[i])
        # img = img2tensor(train_img)
        v_img = img
        p_img = model.forward(v_img)
        loss = mes_loss(p_img, v_img)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('训练损失:', r, train_loss / train_num)

    test_loss = 0
    if r % 3 != 2:
        continue
    model.eval()
    for i in tqdm.tqdm(range(test_num)):
        # test_img = Image.open(test_data_folder + '\\' + test_names[i])
        # img = img2tensor(test_img)
        img = test_set[i].clone()
        v_img = img
        p_img = model.forward(v_img)
        loss = mes_loss(p_img, v_img)
        test_loss += loss.item()
    print('验证损失：', r, test_loss / test_num)
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'self_encoder' + str(test_loss / test_num) + '.pt')
    pass
