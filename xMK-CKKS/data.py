# import imp
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
# import cv2
import numpy as np
import os

# batch_size=64
# transform=tf.Compose([tf.ToTensor(),tf.Normalize([0.1307],[0.3081])])
#normalize正则化，降低模型复杂度，防止过拟合
transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
		])

transform_test = transforms.Compose([
			transforms.RandomCrop(32,padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
		])
# 下载数据集
def get_data():
	train_set=datasets.MNIST(root="./paper/data",train=True,download=True,transform=transform_train)
	test_set=datasets.MNIST(root="./paper/data",train=False,download=True,transform=transform_test)
	train_set1=datasets.CIFAR10(root="./paper/data",train=True,download=True,transform=transform_train)
	test_set1=datasets.CIFAR10(root="./paper/data",train=False,download=True,transform=transform_test)
	return train_set,test_set



# 加载数据集，将数据集变为迭代器
# def get_data_loader():
#     train_loader=DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
#     test_loader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)
#     return train_loader,test_loader

# 显示数据集中的图片

# with open("data/MNIST/raw/train-images-idx3-ubyte","rb") as f:
#     file=f.read()
#     image1=[int(str(item).encode('ascii'),16) for item in file[16:16+784]]
#     image1_np=np.array(image1,dtype=np.uint8).reshape(28,28,1)
#     cv2.imshow("image1_np",image1_np)
#     cv2.waitKey(0)

os.system("pause")