import numpy as np
import tqdm
from 图像信息 import *
import os
import numpy
from PIL import Image

img_names = os.listdir(resize_image_folder)
img_num = len(img_names)
p_train = 0.8
div_list = np.zeros((img_num, 1))
div_list[:int(img_num * p_train)] = 1
np.random.shuffle(div_list)
i = 0
for name in tqdm.tqdm(img_names):
    img = Image.open(resize_image_folder + '\\' + name)
    if div_list[i] == 1:
        img.save(train_data_folder + '\\' + name)
    else:
        img.save(test_data_folder + '\\' + name)
    i += 1

    pass
