import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader




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

# MNIST dataset
def get_data(dataset):

	if dataset=="mnist":
		train_set=datasets.MNIST(root="./paper/data",train=True,download=True,transform=transform_train)
		test_set=datasets.MNIST(root="./paper/data",train=False,download=True,transform=transform_test)
		return train_set,test_set

	elif dataset=="cifar10":
		train_set=datasets.CIFAR10(root="./paper/data",train=True,download=True,transform=transform_train)
		test_set=datasets.CIFAR10(root="./paper/data",train=False,download=True,transform=transform_test)
		return train_set,test_set

