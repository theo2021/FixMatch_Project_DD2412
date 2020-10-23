# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
import argparse
# from torch import Tensor
# from torchvision.transforms import ToTensor
from scheduler.cosineLRreduce import cosineLRreduce
from torch.nn import functional as F
from data.data import DataSet, Augmentation, CustomLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.EMA import EMA
import datetime
import numpy as np
import os

parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--root', type=str, default='~/databases/')              #sudo mkdir /home/databases and then sudo chmod -R 777 /home/databases (for permissions) 
parser.add_argument('--save_dir', type=str, default='~/DeepLearningModels/')
parser.add_argument('--name_model_specs', type=str, default='2020-10-23 17:51:33.557705FixMatchModel_CIFAR10_EMA.state_dict')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='test')

args = parser.parse_args()

def default_collate_fn(ims):
    
    tensors, labels = [torch.FloatTensor(np.array(x[0]).transpose(2, 0, 1))/255 for x in ims], [x[1] for x in ims]

    s = torch.stack(tensors)

    return s, torch.LongTensor(labels)

if __name__ == "__main__":
    from models.wideresnet import WideResNet
    # from models.WideResNet import WideResNet
    import torch
    
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = WideResNet(d=28, k=3, n_classes=10, input_features=3, output_features=16, strides=strides)                # prop unsup/sup: 7 ideal
    model     = WideResNet(3, 28, 2, 10)

    
    # Creating Dataset
    labels_per_class  = [100 for _ in range(10)]
    dataset_loader    = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)
    dataset_loader.load()

    test_data = DataSet(dataset_loader.database, batch = 1)

    test_loader = DataLoader(test_data, collate_fn=default_collate_fn, batch_size=64)

    model_directory = os.path.expanduser(args.save_dir)
    
    model = WideResNet(3, 28, 2, 10)
    state_dict = torch.load(os.path.join(model_directory, args.name_model_specs)) 
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    model.load_state_dict(state_dict)

    correct, total  = 0, 0
    model.eval()
    with torch.no_grad(): #Turn off gradients
        for X, Y in tqdm(test_loader):
            y       = np.argmax(model(X).cpu().numpy(), axis=1) == Y.numpy()
            total  += len(y)
            correct += np.sum(y)

    print('Test accuracy: ', correct/total)
    
