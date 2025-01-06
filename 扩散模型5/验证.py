import os

import torch
from torchvision import transforms
from torch import nn
from 工具类 import load_data, shuffle_and_div_batch, img2tensor, tensor2img, random_noise
from 模型 import auto_encoder
from torch import optim
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plot

log_train = '模型日志/train_log.txt'
log_val = '模型日志/val_log.txt'
train_dict = {'x': [], 'y': []}
test_dict = {'x': [], 'y': []}
with open(log_train, 'r', encoding='utf') as log:
    while True:
        rlog = log.readline()
        if not rlog:
            print('训练日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        train_dict['x'].append(int(rlog[1][:-11]))
        train_dict['y'].append(float(rlog[2][:-1]))
print(train_dict['x'])
print(train_dict['y'])
with open(log_val, 'r', encoding='utf') as log:
    while True:
        rlog = log.readline()

        if not rlog:
            print('验证日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        test_dict['x'].append(int(rlog[1][:-9]))
        test_dict['y'].append(float(rlog[2][:-1]))
plot.plot(train_dict['x'], train_dict['y'])
plot.plot(test_dict['x'], test_dict['y'])
plot.show()

# 下面替换成测试集的地址
f_test = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
          r'\课程项目\archive\TEST')

# 验证集加载,后面是最大加载多少数据
test_data = load_data(f_test, 3000)

# 验证集批次,如果你的测试集小,需要调小这个值
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data2test = shuffle_and_div_batch(test_data, batch_size)
data2test[0] = data2test[0]*random_noise(batch_size)
# data2test[:, :, :, :, ::2] = 0
# data2test[:, :, :, ::2, :] = 0
model = auto_encoder()
model.load_state_dict(torch.load('模型日志/checkpoint_auto_encoder.pt'))
model.eval()
model.to(device)
predict_data, mse_loss, conv_loss = model.forward(data2test[0].to(device), data2test[0].to(device))
print(predict_data.shape)
blackboard = Image.new('RGB', (64 * 2, 64 * batch_size))
for i in range(batch_size):
    blackboard.paste(tensor2img(data2test[0][i]), (64 * 0, 64 * i))
for i in range(batch_size):
    blackboard.paste(tensor2img(predict_data[i]), (64 * 1, 64 * i))
blackboard.show()
