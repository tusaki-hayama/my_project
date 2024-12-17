import torch
from torch import nn, optim
from torchvision import transforms
from model4.配置 import args
from 模型 import auto_encoder
from 工具类 import img2tensor, tensor2img, load_data
import random
#
device = args.device
epochs = args.epoch
batch_size = args.batch_size
study_rare = args.study_rare
f_train_img = args.f_train_img
f_val_img = args.f_val_img
save_model_path = args.save_model_path
use_model2train = args.use_model2train
use_model_path = None
epoch = 0
#
checkpoint = {
    'epoch': 0,
    'model_state_dict': None,
}
#
model = auto_encoder()
if use_model2train:
    use_model_path = args.use_model_path

#
optimizer = optim.Adam(model.parameters(), lr=study_rare)
# 加载数据集
train_tensor = load_data(f_train_img,
                         batch_size, 3, args.img_size, args.img_size,
                         end_flag='训练集加载完成'
                         )
val_tensor = load_data(f_val_img,
                       batch_size, 3, args.img_size, args.img_size,
                       end_flag='验证集加载完成'
                       )





