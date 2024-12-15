from 模型 import coder
import torch
import os
from PIL import Image
from torchvision import transforms
from 配置 import arg
from tqdm import tqdm

arg = arg()
val_data_folder = arg.f_test
val_name = arg.test_names
val_num = arg.test_num
val_model = coder()
val_model.load_state_dict(torch.load('save_model/mseCoder17.805472987992445.pt'))
val_model.eval()
val_model.to(arg.device)
block_noise = torch.ones((49, 3, 28, 28))
for i in range(7):
    for j in range(7):
        block_noise[7 * i + j, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0
line_noise = torch.ones((56, 3, 28, 28))
for i in range(28):
    line_noise[i, :, i, :] = 0
for i in range(28):
    line_noise[28 + i, :, :, i] = 0
big_noise = torch.ones((14 * 14, 3, 28, 28))
for i in range(14):
    for j in range(14):
        big_noise[14 * i + j, :, i:i + 14, j:j + 14] = 0


def add_noise(image):
    background = torch.ones((1, 3, 28, 28))
    index_block = torch.randint(0, 49 - 1, (1, 1))
    index_line = torch.randint(0, 56 - 1, (1, 1))
    index_big = torch.randint(0, 14 * 14 - 1, (1, 1))
    background = background * (block_noise[index_block])[0]
    background = background * (line_noise[index_line])[0]
    background = background * (big_noise[index_big])[0]
    for i in range(1):
        if i > arg.p_noise:
            background[i] = 1
    return image * (background.to(arg.device))
    pass


img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
paper = Image.new('RGB', (28, 28 * 4))
val_img = Image.open(val_data_folder + '//' + val_name[68])
# val_img = Image.open(r'C:\Users\86134\Pictures'
#                       r'\Saved Pictures\HE)K@1D`8LR65UN1R{`_0@G.png').resize((28,28))

v = img2tensor(val_img).view(1, 3, 28, 28)
b_v = add_noise(v.to(arg.device))
b_img = tensor2img(b_v.view(3, 28, 28))
p_v = val_model.forward(b_v)
p_img = tensor2img(p_v.view(3, 28, 28))
p_o_img = tensor2img(val_model.forward(v.to(arg.device)).view(3, 28, 28))
paper.paste(val_img, (0, 0))
paper.paste(b_img, (0, 28))
paper.paste(p_img, (0, 56))
paper.paste(p_o_img, (0, 28 * 3))
paper.show()
