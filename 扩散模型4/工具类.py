from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import os
import torch

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
