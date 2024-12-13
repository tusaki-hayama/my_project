from PIL import Image
from 图像信息 import \
    image_length, \
    image_height, \
    train_folder_address
import os
import tqdm

images_name = os.listdir(train_folder_address)


def resize_image(address, w, h):
    img = Image.open(address)
    img = img.resize((w, h))
    return img


from 图像信息 import resize_image_folder

i = 0
for name in tqdm.tqdm(images_name):
    r_img = resize_image(train_folder_address + '\\' + name, image_length, image_height)
    i += 1
    r_img.save(resize_image_folder + '\\' + str(i) + '.jpg')
