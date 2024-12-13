from 模型 import Coder
import torch
import os
from PIL import Image
from torchvision import transforms

test_data_folder = (r"C:\Users\86134\Desktop\作业\0重修\神经网络深度学习"
                    r"\课程项目\archive\test_images")

test_model = Coder()
test_model.load_state_dict(torch.load('mseCoder5.168201433931561.pt'))
test_model.eval()

test_names = os.listdir(test_data_folder)
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
test_img = Image.open(test_data_folder + '\\' + test_names[59])
# test_img = Image.open(r'C:\Users\86134\Pictures\Saved Pictures\NU7A62B6%T4KA94@8Q`TM{C.png')
# test_img = test_img.resize((28, 28))

test_img_v = img2tensor(test_img).view(1, 3, 28, 28)
# test_img_v = test_img_v.view(1, -1)
predict_img_v = test_model.forward(test_img_v)
predict_img_v = predict_img_v.view(3, 28, 28)
predict_img = tensor2img(predict_img_v)
black_ground = Image.new('RGB', (28, 56))
black_ground.paste(test_img, (0, 0))
black_ground.paste(predict_img, (0, 28))
black_ground.show()
