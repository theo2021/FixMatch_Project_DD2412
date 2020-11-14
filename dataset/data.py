import torchvision
import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
from .ctaugment import CTAugment, apply
import PIL




def fetch_dataset(dbname='CIFAR10', savingdir='~/databases', train = True):
    try:
        db_func = getattr(torchvision.datasets, dbname)
    except AttributeError:
        raise AttributeError("Dataset not in torchvision, check here to find db: https://pytorch.org/docs/stable/torchvision/datasets.html")
    dataset = db_func(savingdir, train, download=True)
    to_memory = []
    mean, var = np.zeros(3), np.zeros(3)
    for sample in dataset:
        to_memory.append(sample)
        mean += np.asarray(sample[0]).mean(axis=0).mean(axis=0)/len(dataset)
        var += (np.asarray(sample[0], dtype=int)**2).mean(axis=0).mean(axis=0)/len(dataset)
    std = np.sqrt((var - mean**2))
    return to_memory, (mean, std)

class default_loader(Dataset):
    def __init__(self, dataset, mean=[125, 125, 125], std=[63,63,63]):
        super().__init__()
        self.dataset = dataset
        self.standard_transform = torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean/255, std/255)])
    
    def __getitem__(self, i):
        im, lbl = self.dataset[i]
        return self.standard_transform(im), lbl

    def __len__(self):
        return len(self.dataset)

class FixMatchDataset(IterableDataset):
    def __init__(self, dataset, labels_per_class, batch_size, mu, augmentation, seed=65, replace=False, mean=[125, 125, 125], std=[63,63,63], ct_update=1, ct_batch=32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.mu = mu
        self.aug = augmentation
        self.ct_update = ct_update
        self.ct_batch = ct_batch
        self.indexes_per_class = []
        # create list if number was given
        if type(labels_per_class) == int:
            indexes = []
            for sample in self.dataset:
                if sample[1] not in indexes:
                    indexes.append(sample[1])
            self.labels_per_class = [labels_per_class for i in range(len(indexes))]
        else:
            self.labels_per_class = labels_per_class
        # indexes for each label
        for label in range(len(self.labels_per_class)):
            sample_list = []
            for i, sample in enumerate(self.dataset):
                if sample[1] == label:
                    sample_list.append(i)
            self.indexes_per_class.append(sample_list)
        self.labeled_indexes = []
        np.random.seed(seed)
        for num_labels, indexes in zip(self.labels_per_class, self.indexes_per_class):
            self.labeled_indexes.extend(np.random.choice(indexes, num_labels, replace=replace))
        self.labeled_indexes = self.labeled_indexes
        self.ulabeled_indexes = list(set(range(len(self.dataset))) - set(self.labeled_indexes))

        self.standard_transform = torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean/255, std/255)])
        self.weak_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomAffine(0, translate=(0.125, 0.125)),
            self.standard_transform])


    def __iter__(self):
        return self

    def __next__(self):
        label_set = np.random.choice(self.labeled_indexes, self.batch_size)
        ulabel_set = np.random.choice(self.ulabeled_indexes, self.mu*self.batch_size)
        l_batch = [None]*self.batch_size
        l_batch_labels = torch.zeros(self.batch_size, dtype=torch.long)
        for i, ind in enumerate(label_set):
            im, lbl = self.dataset[ind]
            l_batch_labels[i] = lbl
            l_batch[i] = self.weak_transform(im)
        l_batch = torch.stack(l_batch)
        output = {'label_samples': l_batch, 'label_targets': l_batch_labels}

        u_batch = [None]*self.batch_size*self.mu
        u_batch_weak = [None]*self.batch_size*self.mu
        for i, ind in enumerate(ulabel_set):
            im, lbl = self.dataset[ind]
            sample_policy = self.aug.policy(False)
            u_batch[i] = self.standard_transform(apply(im, sample_policy))
            u_batch_weak[i] = self.weak_transform(im)
        output['ulabel_samples_strong'] = torch.stack(u_batch)
        output['ulabel_samples_weak'] = torch.stack(u_batch_weak)

        if np.random.choice(self.ct_update) == 0:
            # return ct_update info
            ct_set = np.random.choice(self.labeled_indexes, self.ct_batch)
            ct_batch = [None]*self.ct_batch
            policies = [None]*self.ct_batch
            ct_batch_labels = torch.zeros(self.ct_batch, dtype=torch.long)
            for i, ind in enumerate(ct_set):
                im, lbl = self.dataset[ind]
                ct_batch_labels[i] = lbl
                sample_policy = self.aug.policy(True)
                policies[i] = sample_policy
                ct_batch[i] = self.standard_transform(apply(im, sample_policy))
            ct_batch = torch.stack(ct_batch)
            output['ct_samples'] = ct_batch
            output['policies'] = policies
            output['ct_labels'] = ct_batch_labels

        # pin memory if cuda is availiable
      #  if torch.cuda.is_available():
      #      for key in output.keys():
      #          if torch.is_tensor(output[key]):
      #              output[key].pin_memory()

        return output
        
        # return self.dataset[np.random.randint(len(self.dataset))]

# dataset, (mean, std) = fetch_dataset()
# ctaug = CTAugment()

# db = FixMatchDataset(dataset, 5, 15, 7, mean=mean, std=std, augmentation=ctaug)
# print(0)
# for s in db:
#     print(s)
# print(random)

class FixMatchTestDataset(Dataset):
    def __init__(self, dataset, replace=False, mean=[125, 125, 125], std=[63,63,63], augmentation=False):
        super().__init__()
        self.standard_transform = torchvision.transforms.Compose([
                     torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize(mean/255, std/255)])
        self.weak_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomAffine(0, translate=(0.125, 0.125)),
            self.standard_transform])
        self.dataset = dataset
        self.classes = np.unique([sample[1] for sample in dataset]).shape[0]
        self.aug = augmentation 

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.aug:
            return self.weak_transform(x), y
        else:
            return self.standard_transform(x), y
    
    def __len__(self):
        return len(self.dataset)
        

