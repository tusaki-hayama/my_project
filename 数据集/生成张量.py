import os

import torch
import tqdm
from torchvision import transforms
from PIL import Image
from 图像信息 import *


img2tensor = transforms.ToTensor()
train_name = os.listdir(train_data_folder)[:10]
for name in tqdm.tqdm(train_name):
    pass


