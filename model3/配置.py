import torch
from torch import nn
import os
class arg:
    f_train = (r"C:\Users\86134\Desktop\作业\0重修\神经网络深度学习"
               r"\课程项目\archive\train_images")
    f_test = (r"C:\Users\86134\Desktop\作业\0重修\神经网络深度学习"
              r"\课程项目\archive\test_images")
    epochs = 1000000000
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss(reduction='sum')
    lr = 1e-4
    best_loss = float('inf')
    train_names = os.listdir(f_train)[:]
    train_num = len(train_names)
    test_names = os.listdir(f_test)[:]
    test_num = len(test_names)
    p_noise = 0.5
    model_path = 'save_model/mseCoder2/'
    train_log_path = 'train_log/trainlog.txt'
    test_log_path = 'train_log/testlog.txt'
    pass







