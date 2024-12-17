import os
import random

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from 配置 import args

noise_setting = args.noise_setting
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
batch_size = args.batch_size


def load_data(f_img, batch_size, channel, height, width, end_flag=None):
    img_names = os.listdir(f_img)
    img_num = len(img_names)
    batch_num = img_num // batch_size
    data = torch.zeros((batch_num, batch_size, channel, height, width))
    for i in tqdm(range(batch_num)):
        for j in range(batch_size):
            pil_img = Image.open(f_img + '//' + img_names[batch_size*i+j])
            tensor_img = img2tensor(pil_img)
            data[i][j] = tensor_img
    if end_flag is not None:
        print(end_flag)
    return data
    pass


"""
添加各种噪声:条纹噪声,方块噪声
"""
print('噪声加载器加载中:')
c, h, w = noise_setting['size']
l = noise_setting['len']
r_line1 = torch.ones((l, c, h, w))
r_line2 = torch.ones((l - 1, c, h, w))
r_line4 = torch.ones((l - 2, c, h, w))
r_line8 = torch.ones((l - 3, c, h, w))
c_line1 = torch.ones((l, c, h, w))
c_line2 = torch.ones((l - 1, c, h, w))
c_line4 = torch.ones((l - 2, c, h, w))
c_line8 = torch.ones((l - 3, c, h, w))
####
block1 = torch.ones((l * l, c, h, w))
block2 = torch.ones(((l - 1) ** 2, c, h, w))
block4 = torch.ones(((l - 3) ** 2, c, h, w))
block8 = torch.ones(((l - 2) ** 2, c, h, w))
####
for kind in tqdm(range(l)):
    r_line1[kind, :, kind, :] = 0
for kind in tqdm(range(l - 1)):
    r_line2[kind, :, kind:kind + 2, :] = 0
for kind in tqdm(range(l - 4)):
    r_line4[kind, :, kind:kind + 4, :] = 0
for kind in tqdm(range(l - 8)):
    r_line8[kind, :, kind:kind + 8, :] = 0
for kind in tqdm(range(l)):
    c_line1[kind, :, :, kind] = 0
for kind in tqdm(range(l - 1)):
    c_line2[kind, :, :, kind:kind + 2] = 0
for kind in tqdm(range(l - 4)):
    c_line4[kind, :, :, kind:kind + 4] = 0
for kind in tqdm(range(l - 8)):
    c_line8[kind, :, :, kind:kind + 8] = 0
####
for i in range(l):
    for j in range(l):
        block1[l * i + j, :, i, j] = 0
for i in range(l - 1):
    for j in range(l - 1):
        block2[(l - 1) * i + j, :, i:i + 1, j:j + 1] = 0
for i in range(l - 3):
    for j in range(l - 3):
        block4[(l - 3) * i + j, :, i:i + 3, j:j + 3] = 0
for i in range(l - 7):
    for j in range(l - 7):
        block8[(l - 7) * i + j, :, i:i + 7, j:j + 7] = 0
r_line = torch.cat([r_line1, r_line2, r_line4, r_line8], dim=0)
block = torch.cat([block1, block2, block4, block8], dim=0)
c_line = torch.cat([c_line1, c_line2, c_line4, c_line8], dim=0)
z_noise = torch.cat([r_line, c_line, block], dim=0)
z_noise = z_noise[torch.randperm(z_noise.shape[0])]


# print(r_line.shape)
# print(c_line.shape)
# print(block.shape)
# print(z_noise.shape)
# y = torch.cumprod(z_noise[:batch_size], dim=0)
# print(y.shape)
# tensor2img(y[batch_size - 1]).show()


def add_noise(image):
    global z_noise
    z_noise = z_noise[torch.randperm(z_noise.shape[0])]
    background = torch.cumprod(z_noise[:batch_size], dim=0)
    background = background[torch.randperm(background.shape[0])]
    p_noise = random.randint(1,100)
    for b in range(batch_size):
        if random.randint(1,100)<p_noise:
            background[b] = 1
    return image * background.to(args.device)


# val_tensor = load_data(args.f_val_img,
#                        batch_size, 3, args.img_size, args.img_size,
#                        end_flag='验证集加载完成'
#                        )
# print(val_tensor.shape)
# z_img = add_noise(val_tensor[0])
# blackboard = Image.new('RGB', (64, 64 * batch_size))
# for i in range(batch_size):
#     blackboard.paste(tensor2img(z_img[i]), (0, 64 * i))
# blackboard.show()
