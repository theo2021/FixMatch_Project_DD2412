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
import cv2, random, functools
from collections import defaultdict
from math import ceil
import numpy as np
from .ctaugment import CTAugment, apply
import time
from torch.utils.tensorboard import SummaryWriter

'''
parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--root', type=str, default='/home/firedragon/databases')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)

args = parser.parse_args()
'''

#https://www.cs.toronto.edu/~kriz/cifar.html
#http://ufldl.stanford.edu/housenumbers/

class Augmentation:
    def __init__(self):
        self.cta = CTAugment()
        self.weak = self.__weak()

    def weak_batch(self, x):
        return [(self.weak(im[0]), im[1]) for im in x]
    
    def strong_batch(self, x, probe):

        batch = []

        for img, label in x:
            policy = self.cta.policy(probe)
            batch.append((apply(np.array(img), policy), label, policy))
        return batch

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

class CustomDataLoader(DataLoader):
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



class CustomLoader:

    def __init__(self, labels_per_class, download = True, db_name = 'CIFAR10', db_dir = None, mode = 'train', augmentation = None):
        self.labels_per_class = labels_per_class
        self.db_name = db_name
        self.database = None
        self.db_dir = db_dir
        self.mode = mode
        self.download = download
        self.augmentation = augmentation

        if self.augmentation == None:
            self.augmentation = lambda x : x #set to identity


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
        self.unlabeled_set  = set([x for x in range(len(self))]) - self.labeled_set 

    def __getitem__(self, index):

        image    = self.database[index]
    
        im_pil   = image[0]
        im_label = image[1]

        X, y     = im_pil, im_label
        return X, y

    def __len__(self):
        return len(self.database)

    def visualize(self, index):
        X, y, has_label = self[index]
        X               = cv2.cvtColor(cv2.resize(X.numpy().transpose(1,2,0), (256, 256)), cv2.COLOR_BGR2RGB)
        cv2.imshow('t', X)
        cv2.waitKey()
        #Image.fromarray(X).show()




class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataset, batch=1, steps=None):
        self.database = dataset
        self.steps    = steps
        self.batch    = batch
        self.len      = batch * (steps if steps is not None else len(dataset))
        

    def __getitem__(self, index):
        index   %= len(self.database)
        image    = self.database[index]
    
        im_pil   = image[0]
        im_label = image[1]

        X, y     = im_pil, im_label
        return X, y

    def split(self, p, seed = 42):
        '''
        Split into two datasets given proportion p
        return length p * len(self) and (1-p) * len(self) DISJOINT datasets
        '''
        np.random.seed(seed)
        
        sample_space = [x for x in range(len(self.database))]
        left_ds  = set(random.sample(sample_space, int(p*len(self.database))))
        right_ds = set(sample_space) - left_ds

        return DataSet([self.database[i] for i in left_ds], batch = self.batch, steps = self.steps), DataSet([self.database[i] for i in right_ds], batch = 1, steps = None)


    def __len__(self):
        return self.len

def visualize(img):
    X = cv2.cvtColor(cv2.resize(img.transpose(1,2,0), (256, 256)), cv2.COLOR_BGR2RGB)
    cv2.imshow('t', X)
    cv2.waitKey()


def collate_fn_weak(ims):
    s = augmentation.weak_batch(ims)
    tensors, labels = [x[0] for x in s], [x[1] for x in s]

    return torch.stack(tensors), torch.LongTensor(labels)

def collate_fn_strong(ims):
    # print("previous_rates ", augmentation.cta.rates['autocontrast'][0])
    strong = augmentation.strong_batch(ims)
    weak   = collate_fn_weak(ims)
    
    tensors, labels = [torch.tensor(x[0]) for x in strong], [x[1] for x in strong]
    s = torch.stack(tensors)

    labels, policy = [x[0] for x in labels], labels[0][1]

    return s, weak[0], torch.LongTensor(labels), policy#, augmentation.cta.rates['autocontrast'][0]

'''
if __name__ == "__main__":
   
    labels_per_class  = [10 for _ in range(10)]

    dataset_loader = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)

    dataset_loader.load()
    B = 64
    mu = 7
    K = 2**20
    unlabeled_dataset = DataSet(dataset_loader.get_set(dataset_loader.unlabeled_set), batch=B*mu, steps=K)
    labeled_dataset   = DataSet(dataset_loader.get_set(dataset_loader.labeled_set), batch=B, steps=K)

    augmentation = Augmentation()

    def collate_fn_weak(ims):
        s = augmentation.weak_batch(ims)
        tensors, labels = [x[0] for x in s], [x[1] for x in s]

        return torch.stack(tensors), torch.LongTensor(labels)

    def collate_fn_strong(ims):
        # print("previous_rates ", augmentation.cta.rates['autocontrast'][0])
        strong = augmentation.strong_batch(ims)
        weak   = collate_fn_weak(ims)
        
        tensors, labels = [torch.tensor(x[0]) for x in strong], [x[1] for x in strong]
        s = torch.stack(tensors)

        labels, policy = [x[0] for x in labels], labels[0][1]

        return s, weak[0], torch.LongTensor(labels), policy, augmentation.cta.rates['autocontrast'][0]


    lbl_loader  = DataLoader(labeled_dataset, batch_size = B, collate_fn = collate_fn_weak)
    ulbl_loader = DataLoader(unlabeled_dataset, batch_size = mu*B, collate_fn = collate_fn_strong, num_workers=2)
    # augmentation.cta.rates['autocontrast'][0][2] = 3
    # print(augmentation.cta.rates['autocontrast'])
    # exit()
    start = time.time()
    it=0
    augmentation.cta.depth = 3
    print(len(lbl_loader))
    print(len(ulbl_loader))
    exit()
    # ulbl_loader.collate_fn = collate_fn_strong
    for batch in ulbl_loader:
        #more = batch
        strong, weak, labels, policy, prev = batch
        end = time.time()
        diff = end-start
        time.sleep(2)
        start = time.time()
        
        # print(len(policy), augmentation.cta.depth, end-start)

        it+=1
        
        augmentation.update(policy, -10)
        print("previous_rates ", prev)
        print("seting to ", augmentation.cta.rates['autocontrast'][0])
        print("Time= ", diff)
        # ulbl_loader.collate_fn = collate_fn_strong
        # print("augmentation_depth_set_to ", augmentation.cta.depth)
        
    end = time.time()
    print(end-start)
'''
'''
    #TIME TEST
    start = time.time()
    for (ub, b, lbl) in loader:
        #print(ub, b)
        X = augmentation.strong_batch(ub)
        Y = augmentation.weak_batch(b)
    end = time.time()
    print((((end - start)/loader.K) *2**20)/(60*60), 'h' )
    #print(X)
'''
'''


    
    

    #visualize(augmentation.weak(im).numpy())
    #visualize(augmentation.strong(np.array(im))[0])
    


    #print(augmented_dataset.__getitem__(0))
'''
