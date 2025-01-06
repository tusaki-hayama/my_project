from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import os
import torch
import random
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()


def load_data(f_image, point_num=None):
    image_names = os.listdir(f_image)
    if point_num is None:
        image_nums = len(image_names)
    else:
        image_nums = point_num
    image_Tensor = torch.zeros((image_nums, 3, 64, 64))
    for i in tqdm(range(image_nums), desc='加载数据->'):
        img = Image.open(f_image + '//' + image_names[i])
        image_Tensor[i] = img2tensor(img)
    print('数据加载完毕,大小:{},3,64,64'.format(image_nums))
    return image_Tensor


def shuffle_and_div_batch(image_Tensor, batch_size):
    image_num = image_Tensor.shape[0]
    index = torch.randperm(image_num)
    shuffle_image = image_Tensor[index]
    groups = image_num // batch_size
    div_image = shuffle_image[:groups * batch_size]
    group_batch_image = div_image.view((groups, batch_size, 3, 64, 64))
    return group_batch_image


a = 0.6
b = 0.9
"""
噪声添加:点噪声,线噪声,方块噪声,高斯噪声,组合噪声
"""


def gussi_noise(batch_size):
    lr = 1.5 + random.random()
    noise = torch.randn((batch_size, 1, 64, 64))
    noise[noise <= lr] = 1
    noise[noise > lr] = 0
    for bs in range(batch_size):
        if random.random() > 0.1:
            continue
        noise[bs] = 1
    return noise
    pass


def line_noise(batch_size):
    lr = 1 + random.random()
    noise = torch.ones((batch_size, 1, 64, 64))
    for bs in range(batch_size):
        if random.random() < 0.1:
            continue
        p = torch.randn(64)
        noise[bs, :, p > lr] = 0
        p = torch.randn(64)
        noise[bs, :, :, p > lr] = 0
    return noise
    pass


def block_noise(batch_size):
    noise = torch.ones((batch_size, 1, 64, 64))
    for bs in range(batch_size):
        if random.random() < 0.1:
            continue
        x = random.randint(0, 31)
        y = random.randint(0, 31)
        length = random.randint(8, 32)
        width = random.randint(8, 32)
        noise[bs, :, x:x + width, y:y + length] = 0
    return noise


"""
"""


def random_noise(batch_size):
    noise = torch.ones((batch_size, 1, 64, 64))
    p  = random.random()
    noise = noise * gussi_noise(batch_size) * line_noise(batch_size)* line_noise(batch_size) * block_noise(batch_size)*block_noise(batch_size)
    for bs in range(batch_size):
        if random.random() < p:
            noise[bs] = 1
    return noise
    pass





