import torch
import os
import random
from torchvision import transforms
from torch import nn
from PIL import Image
from tqdm import tqdm

img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()


def load_tensor_data(f_img, batch_size, channel, height, width, end_msg=None):
    img_names = os.listdir(f_img)
    img_num = len(img_names)
    data_tensor = torch.zeros((img_num, channel, height, width))
    load_num = 0
    for name in tqdm(img_names):
        img = Image.open(f_img + '//' + name)
        data_tensor[load_num] = img2tensor(img)
        load_num += 1
    group = img_num // batch_size
    data_num = group * batch_size
    data_tensor = data_tensor[:data_num]
    data_tensor = data_tensor.view((group, batch_size, channel, height, width))
    if end_msg is not None:
        print(end_msg)
    print('数据集加载完成,共有{}个图片数据,截取后有{}个图片数据,数据集大小({},{},{},{},{})'
          .format(img_num, data_num, group, batch_size, channel, height, width))
    return data_tensor


a = 0.6
b = 0.9
"""
噪声添加:点噪声,线噪声,方块噪声,高斯噪声,组合噪声
"""


def gussi_noise(batch_size):
    noise = torch.randn((batch_size, 1, 64, 64))
    noise[noise <= 2.2] = 1
    noise[noise > 2.2] = 0
    return noise
    pass


def line_noise(batch_size):
    noise = torch.ones((batch_size, 1, 64, 64))
    p = torch.randn(64)
    noise[:, :, p > 2.2] = 0
    p = torch.randn(64)
    noise[:, :, :, p > 2.2] = 0
    return noise
    pass



"""
"""


def random_noise(batch_size):
    noise = torch.randn((batch_size, 3, 64, 64)) + torch.randint(-200, 300, (batch_size, 3, 64, 64)) / 100
    noise[noise > 0] = 1
    noise[noise < 0] = 0
    noise[:, 1, :, :] = noise[:, 0, :, :]
    noise[:, 2, :, :] = noise[:, 0, :, :]
    p_noise = a + (b - a) * random.random()
    for bs in range(batch_size):
        if random.random() < p_noise:
            continue
        noise[bs] = 1
    return noise
    pass


noise = random_noise(256)
blackboard = Image.new('RGB', (64 * 64, 64 * 4))
# blackboard.paste(tensor2img(gussi_noise(256)),(0,0))
for i in range(64):
    blackboard.paste(tensor2img(gussi_noise(64)[i]), (64 * i, 0))
    blackboard.paste(tensor2img(line_noise(64)[i]), (64 * i, 64 * 1))
blackboard.show()
