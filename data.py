import os
import argparse
import torchvision
#from tfds.features import FeaturesDict
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
from torchvision import transforms
import cv2


parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--download', type=bool, default=False)
parser.add_argument('--root', type=str, default=os.getcwd()+'databases/')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)

args = parser.parse_args()


#https://www.cs.toronto.edu/~kriz/cifar.html
#http://ufldl.stanford.edu/housenumbers/




class DataSet(torch.utils.data.Dataset):

    def __init__(self, labels_per_class, download = True, db_name = 'CIFAR10', db_dir = None, mode = 'train'):
        self.labels_per_class = labels_per_class
        self.db_name = db_name
        self.database = None
        self.db_dir = db_dir
        self.mode = mode
        self.download = download


    def loader(self):
        print('inside loader')
        if self.db_name == 'CIFAR10':

            if self.mode == 'train':
                self.database = torchvision.datasets.CIFAR10(self.db_dir, train=True, transform=None, target_transform=None, download=self.download)

            elif self.mode == 'test':
                self.database = torchvision.datasets.CIFAR10(self.db_dir, train=False, transform=None, target_transform=None, download=self.download)

        elif self.db_name == 'CIFAR100':

            if self.mode == 'train':
                self.database = torchvision.datasets.CIFAR100(self.db_dir, train=True, transform=None, target_transform=None, download=self.download)

            elif self.mode == 'test':
                self.database = torchvision.datasets.CIFAR100(self.db_dir, train=False, transform=None, target_transform=None, download=self.download)

        elif self.db_name == 'SVHN':

            if self.mode == 'train':
                self.database = torchvision.datasets.SVHN(self.db_dir, split='train', transform=None, target_transform=None, download=self.download)

            elif self.mode == 'test':
                self.database = torchvision.datasets.SVHN(self.db_dir, split='test', transform=None, target_transform=None, download=self.download)

    def __getitem__(self, index):

        print('inside get item')

        image = self.database[index]

        print('debug')

        im_pil = image[0]
        im_label = image[1]


        #transformer_tf = tf.ToTensor()
        X =  torchvision.transforms.ToTensor()(im_pil)
     
        y = torch.tensor(im_label, dtype=torch.long)

        return X, y

    def __len__(self):
        return len(self.database)

    def visualize(self, index):
        X, y = self[index]
        X = cv2.cvtColor(cv2.resize(X.numpy().transpose(1,2,0), (256, 256)), cv2.COLOR_BGR2RGB)
        cv2.imshow('t', X)
        cv2.waitKey()
        #Image.fromarray(X).show()






class Augmentation(DataSet):
    def __init__(self,  download = True, db_name = 'CIFAR10', db_dir = None, mode = 'train', aug_type = 'weak'):
        super().__init__(download, db_name, db_dir, mode)
        self.aug_type = aug_type

    def weak_augment(self):
        transforms     = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.85, 1.15), fillcolor=0),
                        torchvision.transforms.ToTensor()])
        return transforms

    def __getitem__(self, i):
        X, y = self.database[i]#super().__getitem__(i)
        return self.weak_augment()(X), y


#def rand_augment(self, visualize = False):

#def ctaugment(self, visualize = False):


'''
strong_augmentation  = transforms.Compose([
#transforms.Resize(32),  # rescale the image keeping the original aspect ratio
#transforms.CenterCrop(32),  # we get only the center of that rescaled
#transforms.RandomCrop(32),  # random crop within the center crop (data augmentation)
transforms.ColorJitter(brightness=(0.9, 1.1)),
transforms.RandomRotation((-10, 10)),
transforms.RandomHorizontalFlip(),
transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.85, 1.15), fillcolor=0),
# TransformShow(), # visualize transformed pic
transforms.ToTensor()
])

#train_loader = DataLoader(database, batch_size=args.batchsize, shuffle=True, pin_memory=False, num_workers=2)
'''


augmented_dataset = DataSet(db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)

augmented_dataset.loader()
augmented_dataset.visualize(3)

#print(augmented_dataset.__getitem__(0))