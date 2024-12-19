from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import torch
import random

img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()


def load_data(f_img, batch_size, channel, height, width, end_flag=None):
    img_names = os.listdir(f_img)
    img_tensor = torch.zeros((len(img_names), channel, height, width))
    i = 0
    for name in tqdm(img_names):
        img = Image.open(f_img + '//' + name)
        img_tensor[i] = img2tensor(img)
        i += 1
    group = len(img_names) // batch_size
    img_tensor = img_tensor[:group * batch_size]
    img_tensor = img_tensor.view((group, batch_size, channel, height, width))
    if end_flag is not None:
        print(end_flag)
    print('共有图片{}张,返回张量大小:({},{},{},{},{})'.format(group * batch_size, group, batch_size, channel, height,
                                                              width))
    return img_tensor
    pass


def random_noise():
    noise = torch.ones((3, 64, 64))
    x, y = random.randint(0, 32), random.randint(0, 32)
    height = random.randint(16, 32)
    width = random.randint(16, 32)
    noise[:, x:x + height, y:y + width] = 0
    return noise
    pass

# tensor2img(random_noise()).show()