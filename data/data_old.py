import os
import argparse
import torchvision
#from tfds.features import FeaturesDict
import torch
#from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
from torchvision import transforms
import cv2, random, functools
from collections import defaultdict
from math import ceil
import numpy as np
from ctaugment import CTAugment, apply

parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--root', type=str, default=os.getcwd()+'databases/')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)

args = parser.parse_args()


#https://www.cs.toronto.edu/~kriz/cifar.html
#http://ufldl.stanford.edu/housenumbers/

class Augmentation:
    def __init__(self):
        self.cta = CTAugment()
        self.weak = self.__weak()

    def strong(self, x):
        policy = self.cta.policy(True)

        return (apply(x, policy), policy)

    def update(self, policy, accuracy):
        self.cta.update_rates(policy, accuracy)

    def __weak(self):
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                     torchvision.transforms.RandomAffine(0, translate=(0.125, 0.125)),
                     torchvision.transforms.ToTensor()])

        return transforms
class DataLoader:
    def __init__(self, dataset, B, mu, num_classes, K):
        self.dataset      = dataset
        self.B            = B
        self.mu           = mu
        self.num_classes  = num_classes
        self.K = K

    def __iter__(self):
        for i in range(self.K):
            yield self[i]

    def __getitem__(self, i):
        

        unlabled_images = self.dataset.get_set(random.sample(self.dataset.unlabeled_set, self.mu*self.B))
        labeled_images  = self.dataset.get_set(random.sample(self.dataset.labeled_set, self.B))

        unlabeled_batch = [im for im, _, _ in unlabled_images]
        labeled_batch, labels = [im for im, _, _ in labeled_images], [lbl for _, lbl, _ in labeled_images]
        
        return unlabeled_batch, labeled_batch, labels



class DataSet(torch.utils.data.Dataset):

    def __init__(self, labels_per_class, download = True, db_name = 'CIFAR10', db_dir = None, mode = 'train'):
        self.labels_per_class = labels_per_class
        self.db_name = db_name
        self.database = None
        self.db_dir = db_dir
        self.mode = mode
        self.download = download


    def load(self):
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

        self.__init_labels()

    def get_set(self, idxs):
        '''
        idxs a collection of indexes into self
        returns a collection of images
        '''
        return [self[i] for i in idxs]

    def __init_labels(self):
        lbls = defaultdict(list)
        for i,(im, lbl) in enumerate(self.database):
            lbls[lbl].append(i)

        label_set = [random.sample(lbls[lbl], self.labels_per_class[lbl]) for lbl in lbls]

        label_set = set(functools.reduce(lambda a, b : a+b, label_set))

        assert len(label_set) == functools.reduce(lambda a, b : a + b, self.labels_per_class)
        
        self.labeled_set    = label_set
        self.unlabeled_set = set([x for x in range(len(self))]) - self.labeled_set 

    def __getitem__(self, index):
        
        image = self.database[index]
    
        im_pil = image[0]
        im_label = image[1]


        #transformer_tf = tf.ToTensor()
        #X =  torchvision.transforms.ToTensor()(im_pil)
     
        #y = torch.tensor(im_label, dtype=torch.long)
        X, y = im_pil, im_label
        return X, y, index in self.labeled_set

    def __len__(self):
        return len(self.database)

    def visualize(self, index):
        X, y, has_label = self[index]
        X = cv2.cvtColor(cv2.resize(X.numpy().transpose(1,2,0), (256, 256)), cv2.COLOR_BGR2RGB)
        cv2.imshow('t', X)
        cv2.waitKey()
        #Image.fromarray(X).show()





'''
class Augmentation(DataSet):
    def __init__(self,  labels_per_class, download = True, db_name = 'CIFAR10', db_dir = None, mode = 'train', aug_type = 'weak'):
        super().__init__(download, db_name, db_dir, mode)
        self.aug_type = aug_type
    def weak_augment(self):
        transforms     = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.85, 1.15), fillcolor=0),
                        torchvision.transforms.ToTensor()])
        return transforms
    def __getitem__(self, i):
        X, y, has_label = self.database[i]#super().__getitem__(i)
        return self.weak_augment()(X), y, i in self.labeled_set
'''

def visualize(img):
    X = cv2.cvtColor(cv2.resize(img.transpose(1,2,0), (256, 256)), cv2.COLOR_BGR2RGB)
    cv2.imshow('t', X)
    cv2.waitKey()

if __name__ == "__main__":
   
    labels_per_class  = [10 for _ in range(10)]

    dataset = DataSet(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)

    dataset.load()

    loader = DataLoader(dataset, B= 64, mu=7, num_classes=10, K=3)

    for (ub, b, lbl) in loader:
        print(len(ub), len(b), len(lbl))


    
    im = dataset[0][0]
    augmentation = Augmentation()

    visualize(augmentation.weak(im).numpy())
    visualize(augmentation.strong(np.array(im))[0])
    


    #print(augmented_dataset.__getitem__(0))