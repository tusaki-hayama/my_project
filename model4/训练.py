import os

import torch
from torch import nn, optim
from model4.配置 import args
from 模型 import auto_encoder
from 工具类 import img2tensor, tensor2img, load_data, add_noise
from tqdm import tqdm
import random

#
device = args.device
epochs = args.epochs
batch_size = args.batch_size
study_rare = args.study_rare
f_train_img = args.f_train_img
f_val_img = args.f_val_img
save_model_path = args.save_model_path
use_model2train = args.use_model2train
use_model_path = args.use_model_path
best_loss = args.best_loss
mse_loss = args.mse_loss
cross_loss = args.cross_loss
epoch = 0
#
checkpoint = {
    'epoch': 0,
    'model_state_dict': None,
}
#
model = auto_encoder()
model.to(device)
if use_model2train:
    use_model_path = args.use_model_path
    checkpoint = torch.load(use_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

#
optimizer = optim.Adam(model.parameters(), lr=study_rare)
# 加载数据集
train_tensor = load_data(f_train_img,
                         batch_size, 3, args.img_size, args.img_size,
                         end_flag='训练集加载完成'
                         ).to(device)
print('训练集大小,{}'.format(train_tensor.shape))
val_tensor = load_data(f_val_img,
                       batch_size, 3, args.img_size, args.img_size,
                       end_flag='验证集加载完成'
                       ).to(device)
print('验证集大小,{}'.format(val_tensor.shape))
model_list = []
for e in range(epoch, epochs):
    model.train()
    train_loss = 0
    for b in tqdm(range(train_tensor.shape[0]), desc='训练批次中'):
        optimizer.zero_grad()
        img = (train_tensor[b].clone())
        b_img = add_noise(img.to(device))
        p1_img = model.forward(b_img)
        p2_img = model.forward(p1_img)
        p1 = 0.1 + random.random()
        p2 = 0.1 + random.random()
        p3 = 0.1 + random.random()
        p4 = p1 + p2 + p3
        loss = (p1*mse_loss(p1_img, img) + p2*mse_loss(p2_img, img) + p3*mse_loss(p1_img, p2_img))/p4
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('训练轮数{},每批次训练损失:{}'.format(e, train_loss / (train_tensor.shape[0] * train_tensor.shape[1])))
    if e % 3 != 0:
        continue

    model.eval()
    val_loss = 0
    for b in tqdm(range(val_tensor.shape[0]), desc='验证批次中'):
        img = (val_tensor[b].clone())
        b_img = add_noise(img.to(device))
        p1_img = model.forward(b_img)
        p2_img = model.forward(p1_img)
        p1 = 0.1 + random.random()
        p2 = 0.1 + random.random()
        p3 = 0.1 + random.random()
        p4 = p1 + p2 + p3
        loss = (p1 * mse_loss(p1_img, img) + p2 * mse_loss(p2_img, img) + p3 * mse_loss(p1_img, p2_img)) / p4
        val_loss += loss.item()
    print('验证轮数{},每批次验证损失:{}'.format(e, val_loss / (val_tensor.shape[0] * val_tensor.shape[1])))
    if val_loss < best_loss:
        best_loss = val_loss
        checkpoint['epoch'] = e
        checkpoint['model_state_dict'] = model.state_dict()
        model_name = save_model_path + '/checkpointEpoch_{}Loss_{}.pth'.format(e, val_loss / (
                val_tensor.shape[0] * val_tensor.shape[1]))
        model_list.append(model_name)
        torch.save(checkpoint, model_name)
        if len(model_list) > 10:
            del_model = model_list[0]
            model_list.pop(0)
            os.remove(del_model)
