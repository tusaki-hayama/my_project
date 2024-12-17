import numpy as np
from model4.配置 import args
from PIL import Image
from tqdm import tqdm
import os

img_names = os.listdir(args.f_img)
img_num = len(img_names)
print('共有{}张图片'.format(img_num))
l_data = args.l_train + args.l_val + args.l_test
l_train = args.l_train / l_data
l_val = args.l_val / l_data
l_test = args.l_test / l_data
train_num = int(img_num * l_train)
val_num = int(img_num * l_val)
test_num = int(img_num * l_test)
np_l = np.zeros((img_num, 1))
np_l[:train_num] = 1
np_l[train_num:train_num + val_num] = 2
np_l[train_num + val_num:] = 3
np.random.shuffle(np_l)
i = 0
for img_name in tqdm(img_names, desc='预处理'):
    img = Image.open(args.f_img + '\\' + img_name).resize((args.img_size, args.img_size))
    if np_l[i] == 1:
        img.save(args.f_train_img + '\\' + str(i) + '.jpg')
    elif np_l[i] == 2:
        img.save(args.f_val_img + '\\' + str(i) + '.jpg')
    else:
        img.save(args.f_test_img + '\\' + str(i) + '.jpg')
    i = i + 1
    pass
