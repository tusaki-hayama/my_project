import torch
from 模型 import Coder
from 配置 import arg
from torchvision import transforms
from torch import optim
from PIL import Image
from tqdm import tqdm

arg = arg()
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
model = Coder()
model.to(arg.device)
optimizer = optim.Adam(model.parameters(), lr=arg.lr)

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

for epoch in range(arg.epochs):
    model.train()
    train_loss = 0
    for i in tqdm(range(set_train.shape[0])):
        optimizer.zero_grad()
        img = set_train[i].clone()
        p_img = model.forward(img)
        loss = arg.loss(p_img, img)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('训练损失:', epoch, train_loss / arg.train_num)

    test_loss = 0
    if epoch % 10 != 4:
        continue

    model.eval()
    for i in tqdm(range(set_test.shape[0])):
        img = set_test[i].clone()
        p_img = model.forward(img)
        loss = arg.loss(p_img, img)
        test_loss += loss.item()
    print('验证损失:', test_loss / arg.test_num)

    if test_loss < arg.best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'mseCoder' + str(test_loss / arg.test_num) + '.pt')
