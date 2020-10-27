# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
import argparse
# from torch import Tensor
# from torchvision.transforms import ToTensor
from data.ctaugment import transforms as default_transforms
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
parser.add_argument('--root', type=str, default='~/databases/')  
parser.add_argument('--model_directory', type=str, default='testing/')
parser.add_argument('--model_name', type=str, default='CIFAR10_25labels/_final_2020-10-27 01:56:22.910818FixMatchModel_CIFAR10_normal.state_dict')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='test')
parser.add_argument('--save_result', type=str, default='results/fres_25.npy')

args = parser.parse_args()

def default_collate_fn(ims):
    
    tensors, labels = [default_transforms(x[0]) for x in ims], [x[1] for x in ims]

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

    model_directory = os.path.expanduser(args.model_directory)
    
    model = WideResNet(3, 28, 2, 10)
    state_dict = torch.load(os.path.join(model_directory, args.model_name), map_location=torch.device(device)) 
    for k, v in state_dict.items():
        state_dict[k] = v.cpu()
    model.load_state_dict(state_dict)

    correct, total  = 0, 0
    model.eval()
    saving = np.empty((0,2))
    with torch.no_grad(): #Turn off gradients
        for i, (X, Y) in tqdm(enumerate(test_loader)):
            prediction = model(X).softmax(1).numpy()
            confidence = np.max(prediction, axis=1).reshape(-1,1)
            y       = np.argmax(model(X).cpu().numpy(), axis=1) == Y.numpy()
            saving = np.vstack([saving,np.hstack([confidence, y.reshape(-1,1)])])
            total  += len(y)
            correct += np.sum(y)

    print('Test accuracy: ', correct/total)
    np.save(args.save_result, saving, allow_pickle=True)
    
