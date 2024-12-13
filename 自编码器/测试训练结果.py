import os
from 自编码器 import *
from 图像信息 import *
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plot

test_model = self_encoder()
test_model.load_state_dict(torch.load('Coder1560.1636729196161.pt'))
test_model.eval()

test_names = os.listdir(test_data_folder)
img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()
test_img = Image.open(test_data_folder + '\\' + test_names[64])
test_img_v = img2tensor(test_img)
test_img_v = test_img_v.view(1, -1)
predict_img_v = test_model.forward(test_img_v)
predict_img_v = predict_img_v.view(3, 28, 28)
predict_img = tensor2img(predict_img_v)
black_ground = Image.new('RGB', (28, 56))
black_ground.paste(test_img, (0, 0))
black_ground.paste(predict_img, (0, 28))
black_ground.show()
