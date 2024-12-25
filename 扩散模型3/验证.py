import torch
from torch import nn, optim
from torchvision import transforms
from 工具类 import load_tensor_data, img2tensor, tensor2img, random_noise
from 扩散模型 import diffusion_model
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plot

model = diffusion_model()
model.load_state_dict(torch.load('checkpoint_model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
f_test = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
          r'\课程项目\archive\TEST')
batch_size = 64
test_data = load_tensor_data(f_test, batch_size, 3, 64, 64,
                             '测试集加载完成')
p = random.randint(0, test_data.shape[0] - 1)
f_train_log = 'train_log.txt'
f_val_log = 'val_log.txt'
train_dict = {'x': [], 'y': []}
val_dict = {'x': [], 'y': []}
with open(f_train_log,'r',encoding='utf-8') as log:
    while True:
        rlog = log.readline()
        if not rlog:
            print('训练日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        train_dict['x'].append(int(rlog[1][:-11]))
        train_dict['y'].append(float(rlog[2][:-1]))
with open(f_val_log, 'r', encoding='utf') as log:
    while True:
        rlog = log.readline()
        if not rlog:
            print('验证日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        val_dict['x'].append(int(rlog[1][:-10]))
        val_dict['y'].append(float(rlog[2][:-1]))
plot.plot(train_dict['x'], train_dict['y'])
plot.plot(val_dict['x'], val_dict['y'])
plot.show()

batch_img = test_data[p]
blackboard = Image.new('RGB', (64 * batch_size, 64 * 3))
for bs in range(batch_size):
    blackboard.paste(tensor2img(batch_img[bs]), (64 * bs, 64 * 0))
noise = random_noise(batch_size)
noise_img = batch_img * noise
# batch_img[:,:,:,32:60]=0
# noise_img = batch_img
for bs in range(batch_size):
    blackboard.paste(tensor2img(noise_img[bs]), (64 * bs, 64 * 1))
predict_img = model.forward(noise_img.to(device))
for bs in range(batch_size):
    blackboard.paste(tensor2img(predict_img[bs]), (64 * bs, 64 * 2))

blackboard.show()
