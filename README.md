# my_project
kaggle数据集地址:https://www.kaggle.com/datasets/splcher/animefacedataset?resource=download
注意事项:
在文件[扩散模型5]可以找到[验证.py]
将里面的f_test替换成你的测试集的地址,以查看测试结果
否则可以注释掉后面全部,来查看前面的损失曲线.
如果要用来训练模型,需要提供划分好的[训练集]和[验证集]的地址
否则无法运行,以上使用的所有图片都必须是[(3,64,64)的RGB彩色图片]

可以手动创建包含有几张(3,64,64)的图片的文件夹,用f_test指定
注意调节point_num和batch_size,然后就可以运行[验证.py]



