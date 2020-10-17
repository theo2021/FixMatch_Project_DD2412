#This code is inspired by https://github.com/google-research/remixmatch/blob/master/libml/ctaugment.py

import torch
#from data import Dataloader
import torchvision
import tensorflow as tf
import numpy as np

class CTAugment():
    '''
    This method is called during training
    '''

    def __init__(self, nll):
        self.exp_decay_hyperpar = 0.99
        self.nll = nll
        self.transform = None

    def magnitude_bins(self):
        bins = {}

        return bins

    def add_filter(self, pytorch_filter_name):
        filter_dict = {}
        return filter_dict

    def get_filter(self):
        '''

        get filter dictionary to retrive torch filter applied with magnitude
        '''

        filter_set_choice = {}

        return filter_set_choice


    def all_filters(self, params):
        '''
        sample 2 filters from 17 kinds randomly and copose a transformation with that
        '''

        filter_samples = np.random.randint(0,17,2) # 17 kinds of filter

        transform = torchvision.transforms.Compose([
                    self.get_filter[filter_samples[0]],
                    self.get_filter[filter_samples[1]],
                    torchvision.transforms.ToTensor(),
                    ])
        return transform


    def decision(self):


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





