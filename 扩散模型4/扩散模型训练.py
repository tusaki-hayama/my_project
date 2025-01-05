import os
import random
import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img, random_noise
from 自编码器模型 import auto_encoder
from 扩散模型 import diffusion_model
from torch import optim
from tqdm import tqdm
from PIL import Image

epochs = 10000000000
epoch = 0
batch_size = 256
time_step = 50
lr = 1e-1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auto_model = auto_encoder()
auto_model.to(device)
auto_model.load_state_dict(torch.load('自编码器模型/checkpoint_auto_encoder.pt'))
auto_model.eval()
diffusion_model = diffusion_model()
diffusion_model.to(device)
checkpoint_loss = 203
checkpoint_loss = None
checkpoint_model = '自编码器模型/checkpoint_auto_encoder.pt'
checkpoint_model = None
best_loss = checkpoint_loss if checkpoint_loss is not None else float('inf')
mse_loss = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_data(f_train)
print('train_data:{}'.format(train_data.shape))
val_data = load_data(f_val)
print('val_data:{}'.format(val_data.shape))
beta = torch.linspace(0, 1, time_step)
alpha = 1 - beta
alpha_mul = torch.cumprod(alpha, dim=0)
model_list = []
# print(alpha_mul)
while epoch < epochs:
    epoch += 1
    # 准备训练数据
    diffusion_model.train()
    bs = random.randint(64, batch_size)
    data2train = shuffle_and_div_batch(train_data, bs)
    noise_data2train = data2train * random_noise(bs)
    train_loss = 0
    for gs in tqdm(range(data2train.shape[0])):
        v = auto_model.encoder_image(data2train[gs].to(device))
        v_noise = auto_model.encoder_image(noise_data2train[gs].to(device))
        random_t = random.randint(1, time_step - 1)
        X = alpha_mul[random_t] * v_noise + (1 - alpha_mul[random_t]) * torch.randn_like(v_noise) + random_t/time_step
        Y = alpha_mul[random_t - 1] * v + (1 - alpha_mul[random_t - 1]) * (torch.randn_like(v) + v_noise)
        # 训练
        diffusion_model.train()
        X.to(device)
        Y.to(device)
        optimizer.zero_grad()
        predict_Y = diffusion_model.forward(X)
        loss = mse_loss(predict_Y, Y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('第{}轮训练,单向量平均损失为:{}'.format(epoch, train_loss / train_data.shape[0]))
    #
    # break
    if epoch % 3 != 0:
        continue
    diffusion_model.eval()
    val_loss = 0
    data2val = shuffle_and_div_batch(val_data, batch_size)
    noise_data2val = data2val * random_noise(batch_size)
    for gs in tqdm(range(data2val.shape[0])):
        v = auto_model.encoder_image(data2val[gs].to(device))
        v_noise = auto_model.encoder_image(noise_data2val[gs].to(device))
        random_t = random.randint(1, time_step - 1)
        X = alpha_mul[random_t] * v_noise + (1 - alpha_mul[random_t]) * torch.randn_like(v_noise) + random_t/time_step
        Y = alpha_mul[random_t - 1] * v + (1 - alpha_mul[random_t - 1]) * (torch.randn_like(v) + v_noise)
        # 训练
        X.to(device)
        Y.to(device)
        predict_Y = diffusion_model.forward(X)
        loss = mse_loss(predict_Y, Y)
        val_loss += loss.item()
    print('第{}轮验证,单向量平均损失为:{}'.format(epoch, val_loss / val_data.shape[0]))
    if val_loss / val_data.shape[0] < best_loss:
        best_loss = val_loss / val_data.shape[0]
        print('正在保存模型参数')
        torch.save(diffusion_model.state_dict(), '扩散模型/checkpoint_diffusion_encoder.pt')
        model_name = '扩散模型/model_epoch_{}_loss{}.pt'.format(epoch, val_loss / val_data.shape[0])
        torch.save(diffusion_model.state_dict(), model_name)
        model_list.append(model_name)
        print('模型参数保存完成')
        if len(model_list) > 10:
            try:
                del_name = model_list[0]
                model_list.pop(0)
                os.remove(del_name)
            except Exception as e:
                print(e.args)


    pass
