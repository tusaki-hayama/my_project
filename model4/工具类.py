import os

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()


def load_data(f_img, batch_size, channel, height, width, end_flag=None):
    img_names = os.listdir(f_img)
    img_num = len(img_names)
    batch_num = img_num // batch_size
    data = torch.zeros((batch_num, batch_size, channel, height, width))
    for i in tqdm(range(batch_num)):
        for j in range(batch_size):
            pil_img = Image.open(f_img + '//' + img_names[i])
            tensor_img = img2tensor(pil_img)
            data[i][j] = tensor_img
    if end_flag is not None:
        print(end_flag)
    return data
    pass
