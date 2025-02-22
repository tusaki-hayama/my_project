import os

import torch
from torchvision import transforms
from torch import nn


class args:
    # 图像处理部分
    f_img = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
             r'\课程项目\archive\images')
    f_train_img = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
                   r'\课程项目\archive\TRAIN')
    f_val_img = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
                 r'\课程项目\archive\VAL')
    f_test_img = (r'C:\Users\86134\Desktop\作业\0重修\神经网络深度学习'
                  r'\课程项目\archive\TEST')
    l_train = 6
    l_val = 2
    l_test = 2
    img_size = 64
    # 噪声配置
    noise_setting = {'size': (3, 64, 64),'len':64}
    # 训练部分
    use_model2train = True
    use_model_path = '模型保存/模型1/checkpointEpoch_81Loss_677.7348159478635.pth'
    save_model_path = '模型保存/模型1'
    train_log_path = '日志/日志1/train_log.txt'
    val_log_path = '日志/日志1/val_log.txt'
    test_model_name = '模型保存/模型1/checkpointEpoch_81Loss_677.7348159478635.pth'
    epochs = 1000000000
    epoch = 0
    batch_size = 256
    study_rare = 1e-1
    mse_loss = nn.MSELoss(reduction='sum')
    cross_loss = nn.CrossEntropyLoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = float('inf')
    pass
