import torch
from PIL import Image
from torchvision import transforms
from 工具类 import load_data, add_noise
from 配置 import args
from 模型 import auto_encoder

test_data = load_data(args.f_test_img,
                      args.batch_size, 3, args.img_size, args.img_size,
                      end_flag='测试集加载完成'
                      ).to(args.device)

batch_tensor = test_data[1]
tensor2img = transforms.ToPILImage()
checkpoint = torch.load(args.test_model_name)
test_model = auto_encoder()
test_model.load_state_dict(checkpoint['model_state_dict'])
test_model.eval()
test_model.to(args.device)
blackboard = Image.new('RGB', (64 * 4, 64 * args.batch_size))
for b in range(args.batch_size):
    blackboard.paste(tensor2img(batch_tensor[b]), (64 * 0, 64 * b))
predict_tensor = test_model.forward(batch_tensor)
for b in range(args.batch_size):
    blackboard.paste(tensor2img(predict_tensor[b]), (64 * 1, 64 * b))
# batch_tensor = add_noise(batch_tensor)
block = torch.ones_like(batch_tensor)
block[:, :, 8:56, 8:56] = 0
block[:, :, 24:42, 24:42] = 1
batch_tensor = batch_tensor * block
for b in range(args.batch_size):
    blackboard.paste(tensor2img(batch_tensor[b]), (64 * 2, 64 * b))
predict_tensor = test_model.forward(batch_tensor)
for b in range(args.batch_size):
    blackboard.paste(tensor2img(predict_tensor[b]), (64 * 3, 64 * b))
blackboard.show()
