import random

from 模型 import coder
import torch
import os
import re
import matplotlib.pyplot as plot
from PIL import Image
from torchvision import transforms
from 配置 import arg
from tqdm import tqdm

arg = arg()
val_data_folder = arg.f_test
val_name = arg.test_names
val_num = arg.test_num
val_model = coder()
val_model.load_state_dict(torch.load('save_model/mseCoder4/mseModel10.232735732712985.pt'))
val_model.eval()
val_model.to(arg.device)
train_log_name = arg.train_log_path
test_log_name = arg.test_log_path
train_color = (255, 0, 255)
test_color = (255, 255, 0)
train_dict = {'x': [], 'y': []}
test_dict = {'x': [], 'y': []}
with open(arg.train_log_path, 'r', encoding='utf') as log:
    while True:
        rlog = log.readline()
        if not rlog:
            print('训练日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        train_dict['x'].append(int(rlog[1][:-11]))
        train_dict['y'].append(float(rlog[2][:-1]))
with open(arg.test_log_path, 'r', encoding='utf') as log:
    while True:
        rlog = log.readline()
        if not rlog:
            print('验证日志处理完毕')
            break
        rlog = rlog.split(':')
        # print(rlog)
        test_dict['x'].append(int(rlog[1][:-10]))
        test_dict['y'].append(float(rlog[2][:-1]))
plot.plot(train_dict['x'], train_dict['y'])
plot.plot(test_dict['x'], test_dict['y'])
plot.show()

block_noise = torch.ones((49, 3, 28, 28))
for i in range(7):
    for j in range(7):
        block_noise[7 * i + j, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0
line_noise = torch.ones((56, 3, 28, 28))
for i in range(28):
    line_noise[i, :, i, :] = 0
for i in range(28):
    line_noise[28 + i, :, :, i] = 0
big_noise = torch.ones((14 * 14, 3, 28, 28))
for i in range(14):
    for j in range(14):
        big_noise[14 * i + j, :, i:i + 14, j:j + 14] = 0


def add_noise(image):
    background = torch.ones((1, 3, 28, 28))
    index_block = torch.randint(0, 49 - 1, (1, 1))
    index_line = torch.randint(0, 56 - 1, (1, 1))
    index_big = torch.randint(0, 14 * 14 - 1, (1, 1))
    background = background * (block_noise[index_block])[0]
    background = background * (line_noise[index_line])[0]
    background = background * (big_noise[index_big])[0]
    # print(background.shape)
    # for i in range(1):
    #     break
    #     if 1 > arg.p_noise:
    #         background[i] = 1
    return image * (background.to(arg.device))
    pass


img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
check_num = 128
paper = Image.new('RGB', (28 * check_num, 28 * 4))
check_v = torch.zeros((check_num, 3, 28, 28))
check_img = []
print('加载测试子集')
for i in tqdm(range(check_num)):
    check_img.append(Image.open(arg.f_test + '//' + arg.test_names[random.randint(0, arg.test_num - 1)]))
    check_v[i] = img2tensor(check_img[i])
print('打印原始图片')
for i in tqdm(range(check_num)):
    paper.paste(check_img[i], (28 * i, 28*0))
noise_v = add_noise(check_v.to(arg.device))
noise_img = []
for i in tqdm(range(check_num)):
    noise_img.append(tensor2img(noise_v[i]))
print('打印遮掩图片')
for i in tqdm(range(check_num)):
    paper.paste(noise_img[i], (28 * i, 28*2))
print('打印预测图片')
p_v = val_model.forward(noise_v)
p_img = []
for i in tqdm(range(check_num)):
    p_img.append(tensor2img(p_v[i]))
for i in tqdm(range(check_num)):
    paper.paste(p_img[i], (28 * i, 28*3))
print('原始图片重建')
r_v = val_model.forward(check_v.to(arg.device))
r_img = []
for i in tqdm(range(check_num)):
    r_img.append(tensor2img(r_v[i]))
for i in tqdm(range(check_num)):
    paper.paste(r_img[i], (28 * i, 28*1))
paper.show()
