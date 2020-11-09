import torch
from models.wideresnet import WideResNet
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from dataset.data import fetch_dataset, FixMatchTestDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
import os

parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('--db_dir', type=str, default='~/databases/')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--model_dir', type=str, default='~/new_models_tst/final_ema.state_dict')
parser.add_argument('--save_result', type=str, default='~/new_models_tst/results/fres_25.npy')

args = parser.parse_args()

_, (mean, std) = fetch_dataset(dbname=args.dataset, savingdir=args.db_dir, train = True)
print("mean, std calculated")
dataset, _ = fetch_dataset(dbname=args.dataset, savingdir=args.db_dir, train = False)
test_dataset = FixMatchTestDataset(dataset, mean=mean, std=std)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("loading model")
model = WideResNet(3, 28, 2, test_dataset.classes)
model_directory = os.path.expanduser(args.model_dir)
    
state_dict = torch.load(model_directory, map_location=device)
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
save_dir = os.path.expanduser(args.save_result)
np.save(save_dir, saving, allow_pickle=True)