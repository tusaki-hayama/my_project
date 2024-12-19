import torch
from torch import nn, optim
from torchvision import transforms
from 工具类 import load_data, img2tensor, tensor2img, random_noise
from 扩散模型 import diffusion_model
from tqdm import tqdm
import random

max_step = 10
batch_size = 64
study_rare = 1e-8
epochs = 1000000000
l = torch.linspace(0, 1, steps=max_step)
print('步幅:{}'.format(l))

f_train = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
           r'\课程项目\archive\TRAIN')
f_val = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
         r'\课程项目\archive\VAL')
train_data = load_data(f_train, batch_size, 3, 64, 64,
                       '训练集加载完成')
val_data = load_data(f_val, batch_size, 3, 64, 64,
                     '测试集加载完成')

model = diffusion_model()
optimizer = optim.Adam(model.parameters(), lr=study_rare)
mse_loss = nn.MSELoss(reduction='sum')
for epoch in range(epochs):
    model.train()
    # 抽一张图片
    sample_img = train_data[random.randint(0, 595 - 1)][random.randint(0, 64 - 1)]
    # 抽取一个噪声强度和降级噪声强度
    noise_level = random.randint(1, 9)
    noise_down_level = noise_level - 1
    # 抽取一个噪声掩码
    noise_mask = random_noise()
    # 生成噪声图
    noise_img = sample_img * noise_mask * l[noise_level] + sample_img * (1 - l[noise_level])
    # 生成降噪图
    down_noise_img = sample_img * noise_mask * l[noise_down_level] + sample_img * (1 - l[noise_down_level])
    # 模型预测
    noise_img = noise_img.view((1, 3, 64, 64))
    down_noise_img = down_noise_img.view((1, 3, 64, 64))
    b_steps = torch.zeros((1, 2))
    b_steps[:, 0] = noise_level
    b_steps[:, 1] = noise_down_level
    optimizer.zero_grad()
    re_image = model.forward(noise_img, b_steps)
    train_loss = mse_loss(re_image, down_noise_img)
    train_loss.backward()
    optimizer.step()
    print('第{}次训练,噪声强度{},训练损失{}'.format(epoch, noise_level, train_loss.item()))
    if epoch % 10000 == 1000:
        torch.save(model.state_dict(), 'first_model.pt')
    pass
