import torch
from dataset.data import FixMatchDataset, fetch_dataset, default_loader, collate_wrapper
from dataset.ctaugment import CTAugment
from torch.utils.data import DataLoader
import argparse
from sklearn.model_selection import train_test_split
from time import time
from models.wideresnet import WideResNet
from schedulers.cosineLRreduce import cosineLRreduce
from torch.utils.tensorboard import SummaryWriter
from train import train_fixmatch
from models.ema import EMA
import os


parser = argparse.ArgumentParser(description='FixMatch_Model')
parser.add_argument('--db_dir', type=str, default='~/databases/')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--labels_per_class', type=int, default=4)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--mu', type=int, default=7)
parser.add_argument('--ct_augment_update', type=int, default=10)
parser.add_argument('--ct_augment_batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=6)
parser.add_argument('--validation_percentage', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=65)
parser.add_argument('--steps', type = int, default=2**20)
parser.add_argument('--scheduler_steps', type = int, default=2**20)
parser.add_argument('--warmup_scheduler', type = float, default=0.01) #none, will to 1% of training
parser.add_argument('--lr', type= float, default=0.03)
parser.add_argument('--momentum', type= float, default=0.9)
parser.add_argument('--nesterov', type= bool, default=True)
parser.add_argument('--weight_decay', type= float, default=0.0005)
parser.add_argument('--model_save_dir', type=str, default="FixMatch_models")
parser.add_argument('--tb_logdir', type= str, default="tensorboard_log")
parser.add_argument('--loss_func', type=str, default="cross,cross")
parser.add_argument('--gcross_thresh', type= float, default=0.7)
parser.add_argument('--ema', type=bool, default=True)
parser.add_argument('--sheduler_from_step', type=int, default=0)
parser.add_argument('--load_model', type=str, default='')
parser.add_argument('--cta_load', type=str, default='')
parser.add_argument('--loss_lambda', type=int, default=1)
args = parser.parse_args()

# create folders if they don't exist
for folder in [args.model_save_dir, args.tb_logdir]:
    if not os.path.exists(folder):
        os.makedirs(folder)
model_save_dir = os.path.expanduser(args.model_save_dir)
tb_logdir = os.path.expanduser(args.tb_logdir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset, (mean, std) = fetch_dataset(dbname=args.dataset, savingdir=os.path.expanduser(args.db_dir), train = True)
print('Dataset mean & std', mean, std)
ctaug = CTAugment(model_save_dir) # Augmentation object
if os.path.isfile(os.path.expanduser(args.cta_load)):
    print("Loading CTA")
    ctaug.load_rates(os.path.expanduser(args.cta_load))
dataset_train, dataset_val = train_test_split(dataset, test_size=args.validation_percentage, random_state=args.seed)
print('Train / Validation Set', len(dataset_train), len(dataset_val))
# Datasets
db_object = FixMatchDataset(dataset_train, labels_per_class=args.labels_per_class, batch_size=args.batch_size, mu=args.mu, seed=args.seed, replace=False, mean=mean, std=std, augmentation=ctaug, ct_update=args.ct_augment_update, ct_batch=args.ct_augment_batch_size)
db_val = default_loader(dataset_val,  mean=mean, std=std)
# Dataloaders
train_loader = DataLoader(db_object, batch_size=None, num_workers=args.workers, collate_fn=collate_wrapper)
val_loader = DataLoader(db_val, batch_size=args.batch_size, num_workers=2)

# model, optimizer, scheduler
model     = WideResNet(3, 28, 2, len(db_object.labels_per_class)).to(device)
model_directory = os.path.expanduser(args.load_model)
if os.path.isfile(model_directory):
    print('Loading model ', model_directory)
    state_dict = torch.load(model_directory, map_location=device)
    model.load_state_dict(state_dict)

if args.ema:
    ema = EMA(model)
else:
    ema=None
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = cosineLRreduce(optimizer, args.scheduler_steps, warmup=args.warmup_scheduler, from_step=args.sheduler_from_step)
K = args.steps

# TensorBoard log
tb_writer = SummaryWriter(log_dir=tb_logdir)
train_fixmatch(train_loader, val_loader, model, K, ctaug, optimizer, scheduler,device, tb_writer, threshold = args.threshold, ema=ema, saving_dir=model_save_dir, lqloss_tresh=args.gcross_thresh, loss_functions=args.loss_func, loss_lambda=args.loss_lambda)
