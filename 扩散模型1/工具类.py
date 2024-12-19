from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
import torch

img2tensor = transforms.ToTensor()


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
    print('返回张量大小:({},{},{},{},{})'.format(group, batch_size, channel, height, width))
    return img_tensor
    pass
