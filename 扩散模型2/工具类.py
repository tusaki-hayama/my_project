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


a = 0.7
b = 0.9


def random_noise(batch_size):
    noise = torch.ones((batch_size, 3, 64, 64))
    p_noise = a + (b - a) * random.random()
    for bs in range(batch_size):
        if random.random() > p_noise:
            continue
        x, y = random.randint(0, 16), random.randint(0, 16)
        height = random.randint(32, 48)
        width = random.randint(32, 48)
        noise[bs, :, x:x + height, y:y + width] = 0
    return noise
    pass

# noise = random_noise(64)
# blackboard = Image.new('RGB', (64 * 64, 64))
# for i in range(64):
#     blackboard.paste(tensor2img(noise[i]), (64 * i, 0))
# blackboard.show()
