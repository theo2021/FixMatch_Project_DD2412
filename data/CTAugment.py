#This code is inspired by https://github.com/google-research/remixmatch/blob/master/libml/ctaugment.py



import torch
#from data import Dataloader
import torchvision
import tensorflow as tf
import numpy as np

class CTAugment:
    '''
    This method is called during training
    '''

    def __init__(self, depth = 1, num_filters = 2):
        self.transform = None
        self.filter_samples = None
        self.depth = depth
        self.exp_decay_hyperpar = 0.99
        self.m = np.ones(num_filters)
        self.filter_dict = {}
        self.bins = np.zeros(num_filters)

    def decay_hyperpar_anneal(self, update_step, total_steps, a = 1):
        self.exp_decay_hyperpar *=  np.exp(-a*(update_step/total_steps))

    def magnitude_bins(self, loss):
        self.bins = self.exp_decay_hyperpar * self.m + (1 - self.exp_decay_hyperpar) * loss

    def add_filter(self):
        self.filter_dict = {
            0 : torchvision.transforms.RandomHorizontalFlip(),
            1 : torchvision.transforms.RandomAffine(translate=(0.125, 0.125))
        }

    def all_filter(self):
        '''
        get filter dictionary to retrive torch filter applied with magnitude
        '''

        chosen_filters = []
        for idx in range(len(self.filter_samples)):
            categ = self.filter_samples[idx]
            if categ > 0:
                for idx in range(categ):
                    chosen_filters.append(
                        self.filter_dict[idx](
                            self.bins
                        )
                    )
        return chosen_filters

    def get_filters(self):
        '''
        sample 2 filters from 17 kinds randomly and compose a transformation with that
        '''

        self.filter_samples = None

        m_hat = self.bins * (self.bins > 0.8)

        self.filter_samples = np.random.multinomial(n=self.depth, pvals = m_hat/np.sum(m_hat))

        self.transform = torchvision.transforms.Compose([
                    self.all_filter[0],
                    self.all_filter[1],
                    torchvision.transforms.ToTensor(),
                    ])

    def apply_aug(self, input_PIL_img):
        '''
        :param input_PIL_img: the input image must be transformed from tensor to PIL befor applying CTAugmentation
        :param transform: choose a set of transformations to apply
        :return: return tensor image ready to go to cuda
        '''

        if self.transform:
            X = self.transform(input_PIL_img)
        else:
            transformer_tf = tf.ToTensor()
            X = transformer_tf(input_PIL_img)
        return X


### EXAMPLE USE

#ctaugment = CTAugment()
#ctaugment.decay_hyperpar_anneal(<actual timestep> , <total timestep>)
#ctaugment.magnitude_bins(<loss from train>)
#ctaugment.add_filter()
#ctaugment.get_filters()
#ctaugment.apply_aug(<desired image to be augmented>)


