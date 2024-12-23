import torch
from torch import nn, optim
from torchvision import transforms
from 工具类 import load_tensor_data, img2tensor, tensor2img, random_noise
from 扩散模型 import diffusion_model
from tqdm import tqdm
import random
from PIL import Image

model = diffusion_model()
model.load_state_dict(torch.load('checkpoint_model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
f_test = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
          r'\课程项目\archive\TEST')
batch_size = 64
test_data = load_tensor_data(f_test, batch_size, 3, 64, 64,
                             '测试集加载完成')
p = random.randint(0, test_data.shape[0] - 1)
batch_img = test_data[p]
blackboard = Image.new('RGB', (64 * batch_size, 64 * 3))
for bs in range(batch_size):
    blackboard.paste(tensor2img(batch_img[bs]), (64 * bs, 64 * 0))
noise = random_noise(batch_size)
noise_img = batch_img * noise
for bs in range(batch_size):
    blackboard.paste(tensor2img(noise_img[bs]), (64 * bs, 64 * 1))
predict_img = model.forward(noise_img.to(device))
for bs in range(batch_size):
    blackboard.paste(tensor2img(predict_img[bs]), (64 * bs, 64 * 2))

blackboard.show()
