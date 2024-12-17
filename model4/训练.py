import torch
from torch import nn, optim
from torchvision import transforms
from model4.配置 import args
from 模型 import auto_encoder
from 工具类 import img2tensor, tensor2img, load_data, add_noise
from tqdm import tqdm
import random

#
device = args.device
epochs = args.epoch
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

#
optimizer = optim.Adam(model.parameters(), lr=study_rare)
# 加载数据集
train_tensor = load_data(f_train_img,
                         batch_size, 3, args.img_size, args.img_size,
                         end_flag='训练集加载完成'
                         )
print('训练集大小,{}'.format(train_tensor.shape))
val_tensor = load_data(f_val_img,
                       batch_size, 3, args.img_size, args.img_size,
                       end_flag='验证集加载完成'
                       )
print('验证集大小,{}'.format(val_tensor.shape))

for e in tqdm(range(epochs)):
    model.train()
    train_loss = 0
    print(222)
    for b in range(297):
        optimizer.zero_grad()
        img = train_tensor[b].clone()
        b_img = add_noise(train_tensor[b])
        p1_img = model.forward(b_img)
        p2_img = model.forward(p1_img)
        loss = mse_loss(p1_img, img) + mse_loss(p2_img, img)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('训练轮数{},每批次训练损失:{}'.format(epoch, train_loss / train_tensor.shape[0]))
print(1111)
