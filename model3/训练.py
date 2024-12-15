import os
import torch
from 模型 import coder
from 配置 import arg
from torchvision import transforms
from torch import optim
from PIL import Image
from tqdm import tqdm
import random

arg = arg()
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
model = coder()
model.to(arg.device)
optimizer = optim.Adam(model.parameters(), lr=arg.lr)

block_noise = torch.ones((49, 3, 28, 28))
for i in range(7):
    for j in range(7):
        block_noise[7 * i + j, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = 0
# black_ground = Image.new('RGB', (28 * 7, 28 * 7))
# for i in range(7):
#     for j in range(7):
#         black_ground.paste(tensor2img(block_noise[7 * i + j]), (28 * i, 28 * j))
# black_ground.show()
line_noise = torch.ones((56, 3, 28, 28))
for i in range(28):
    line_noise[i, :, i, :] = 0
for i in range(28):
    line_noise[28 + i, :, :, i] = 0
# black_ground = Image.new('RGB', (28 * 14, 28 * 4))
# for i in range(14):
#     for j in range(4):
#         black_ground.paste(tensor2img(line_noise[4 * i + j]), (28 * i, 28 * j))
# black_ground.show()
big_noise = torch.ones((14 * 14, 3, 28, 28))
for i in range(14):
    for j in range(14):
        big_noise[14 * i + j, :, i:i + 14, j:j + 14] = 0


# black_ground = Image.new('RGB', (28 * 14, 28 * 14))
# index = torch.randperm(14*14)
# big_noise = big_noise[index]
# for i in range(14):
#     for j in range(14):
#         black_ground.paste(tensor2img(big_noise[14 * i + j]), (28 * i, 28 * j))
# black_ground.show()


def add_noise(image):
    background = torch.ones((arg.batch_size, 3, 28, 28))
    index_block = torch.randint(0, 49 - 1, (1, arg.batch_size))
    index_line = torch.randint(0, 56 - 1, (1, arg.batch_size))
    index_big = torch.randint(0, 14 * 14 - 1, (1, arg.batch_size))
    background = background * (block_noise[index_block])[0]
    background = background * (line_noise[index_line])[0]
    background = background * (big_noise[index_big])[0]
    p_noise = random.random()
    for i in range(arg.batch_size):
        if random.random() > p_noise:
            background[i] = 1
    return image * (background.to(arg.device))
    pass


# 加载训练集
print('加载训练集')
set_train = torch.zeros((arg.train_num, 3, 28, 28))
for i in tqdm(range(arg.train_num)):
    img = Image.open(arg.f_train + '\\' + arg.train_names[i])
    v_img = img2tensor(img)
    set_train[i] = v_img
set_train = set_train[:(arg.train_num // arg.batch_size) * arg.batch_size]
set_train = set_train.view(arg.train_num // arg.batch_size, arg.batch_size, 3, 28, 28).to(arg.device)
# 加载测试集
print('加载测试集')
set_test = torch.zeros((arg.test_num, 3, 28, 28))
for i in tqdm(range(arg.test_num)):
    img = Image.open(arg.f_test + '\\' + arg.test_names[i])
    v_img = img2tensor(img)
    set_test[i] = v_img
set_test = set_test[:(arg.test_num // arg.batch_size) * arg.batch_size]
set_test = set_test.view(arg.test_num // arg.batch_size, arg.batch_size, 3, 28, 28).to(arg.device)
save_file = []
for epoch in range(arg.epochs):
    model.train()
    train_loss = 0
    for i in tqdm(range(set_train.shape[0])):
        optimizer.zero_grad()
        img = set_train[i].clone()
        b_img = add_noise(set_train[i]).clone()
        p_img = model.forward(b_img)
        e_img = model.forward(p_img)
        p1 = 0.1+random.random()
        p2 = 0.1+random.random()
        p3 = 0.1+random.random()
        p4 = p1+p2+p3
        loss = (p1*arg.loss(e_img, img) + p2*arg.loss(e_img, p_img) + p3*arg.loss(p_img,img))/p4
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('训练损失:', epoch, train_loss / arg.train_num)
    with open(arg.train_log_path, 'a+', encoding='utf') as log:
        log.write('epoch:{},train_loss:{}\n'.
                  format(epoch, train_loss / arg.train_num))
    test_loss = 0
    if epoch % 10 != 3:
        continue

    model.eval()
    for i in tqdm(range(set_test.shape[0])):
        img = set_test[i].clone()
        b_img = add_noise(set_test[i]).clone()
        p_img = model.forward(b_img)
        e_img = model.forward(p_img)
        p1 = 0.1 + random.random()
        p2 = 0.1 + random.random()
        p3 = 0.1 + random.random()
        p4 = p1 + p2 + p3
        loss = (p1*arg.loss(e_img, img) + p2*arg.loss(e_img, p_img) + p3*arg.loss(p_img,img))/p4
        test_loss += loss.item()
    print('验证损失:', test_loss / arg.test_num)
    with open(arg.test_log_path, 'a+', encoding='utf') as log:
        log.write('epoch:{},test_loss:{}\n'.
                  format(epoch, test_loss / arg.test_num))
    if test_loss < arg.best_loss:
        arg.best_loss = test_loss
        file_address = arg.model_path + 'mseModel' + str(test_loss / arg.test_num) + '.pt'
        save_file.append(file_address)
        torch.save(model.state_dict(), file_address)
        if len(save_file) > 10:
            del_file = save_file[0]
            save_file.pop(0)
            os.remove(del_file)
