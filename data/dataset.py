## this file is to download the datasets for the projects: CIFAR-10, CIFAR-100, SVHN

import torchvision

root = 'data/'

CIFAR10_train = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
CIFAR10_test = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=False)

print(CIFAR10_train.shape)

#CIFAR100_train = torchvision.datasets.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
#CIFAR100_test = torchvision.datasets.CIFAR100(root, train=False, transform=None, target_transform=None, download=False)

#SVHN_train = torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
#SVHN_test = torchvision.datasets.SVHN(root, split='test', transform=None, target_transform=None, download=False)

