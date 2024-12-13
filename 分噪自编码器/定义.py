import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as func
from 图像信息 import *
import os
import tqdm
from PIL import Image


class recoder(nn.Module):
    def __init__(self):
        super(recoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 * 3, 176),
            nn.BatchNorm1d(176),
            nn.ReLU(),
            nn.Linear(176, 30),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 176),
            nn.BatchNorm1d(176),
            nn.ReLU(),
            nn.Linear(176, 28 * 28 * 4),
        )
        pass

    def forward(self, input_x):
        x = self.encoder(input_x)
        x = self.decoder(x)
        return x
        pass

    pass


img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mes_loss = nn.CrossEntropyLoss(reduction='sum')
model_name = '自编码器.pt'
epochs = 10000000
batch_size = 16
lr = 1e-5

train_names = os.listdir(train_data_folder)[:]
train_num = len(train_names)
test_names = os.listdir(test_data_folder)[:]
test_num = len(test_names)
best_loss = float('inf')

# 读取训练集
print('加载训练集')
train_set = torch.zeros((train_num, 3, 28, 28))
for i in tqdm.tqdm(range(train_num)):
    img = Image.open(train_data_folder + '\\' + train_names[i])
    img_v = img2tensor(img)
    train_set[i] = img_v
train_set = train_set[:(train_num // batch_size) * batch_size]
train_set = train_set.view(train_num // batch_size, batch_size, 3, 28, 28).to(device)

# 读取测试集
print('加载测试集')
test_set = torch.zeros((test_num, 3, 28, 28))
for i in tqdm.tqdm(range(test_num)):
    img = Image.open(test_data_folder + '\\' + test_names[i])
    img_v = img2tensor(img)
    test_set[i] = img_v
test_set = test_set[:(test_num // batch_size) * batch_size]
test_set = test_set.view(test_num // batch_size, batch_size, 3, 28, 28).to(device)



