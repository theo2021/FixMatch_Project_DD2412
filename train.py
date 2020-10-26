# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
import argparse
# from torch import Tensor
# from torchvision.transforms import ToTensor
from scheduler.cosineLRreduce import cosineLRreduce
from torch.nn import functional as F
from data.data import DataSet, Augmentation, CustomLoader
from data.ctaugment import cutout
from data.ctaugment import transforms as default_transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.EMA import EMA
import functools
import datetime
import numpy as np
import os

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--tbsw_logdir', type=str, default=None)
parser.add_argument('--root', type=str, default='~/databases/')              #sudo mkdir /home/databases and then sudo chmod -R 777 /home/databases (for permissions) 
parser.add_argument('--save_dir', type=str, default='~/DeepLearningModels/')
parser.add_argument('--name_model_specs', type=str, default='FixMatchModel')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--validation_split', type=float, default=0.9)
parser.add_argument('--task', type=str, default='train')

# parsed in main
parser.add_argument('--B', type = int, default = 32)     # 64
parser.add_argument('--K', type = int, default = 1000)    # 220
parser.add_argument('--mu', type = int, default = 7)     # 7
parser.add_argument('--labels_per_class', type = int, default = 100)
parser.add_argument('--confidence_threshold', type = int, default = 15)
parser.add_argument('--warmup_scheduler', type = float, default=0.05) #none, will to 1% of training
parser.add_argument('--lr', type= float, default=0.03)
parser.add_argument('--momentum', type= float, default=0.9)
parser.add_argument('--nesterov', type= bool, default=True)
parser.add_argument('--weight_decay', type= float, default=0.0005)
parser.add_argument('--num_workers1', type=int, default=1)
parser.add_argument('--num_workers2', type=int, default=2)
parser.add_argument('--num_workers3', type=int, default=3)


# To Do
parser.add_argument('--UKF_noise_regularizer', type=bool, default=False)
parser.add_argument('--ensemble_model', type=bool, default=False)



args = parser.parse_args()

augmentation  = Augmentation()


class fixmatch_Loss():
    
    def __init__(self, l=1, threshold=0.95):
        self.u_weight          = l
        self.tau               = threshold

    def get_pseudo(self, unlabeled_weak_predictions):
        max_pred, pseudolabels = unlabeled_weak_predictions.softmax(1).max(axis=1)
        indexes_over_threshold = max_pred > self.tau
        return indexes_over_threshold, pseudolabels

    def calculate_losses(self, labeled_prediction, labeled_labels, unlabeled_strong_predictions, pseudolabels, usize):
        supervised = F.cross_entropy(labeled_prediction, labeled_labels)
        if pseudolabels.size()[0] > 0:
            unsupervised = F.cross_entropy(unlabeled_strong_predictions, pseudolabels, reduction='sum')/usize
        else:
            unsupervised = 0
        return supervised + self.u_weight*unsupervised, supervised, unsupervised



def train_fixmatch(model, ema, trainloader, validation_loader, augmentation, optimizer, scheduler, device, K, tb_writer):
    # if model is confident for this threshold start the unlabeled and ctaugment
    lossfunc                 = fixmatch_Loss()
    run_validation           = 200
    run_ctaugment            = 6
    update_bar = 50
    top_val = 0
    end_warmup = False
    with tqdm(total = K) as bar:
        itertrain, iterval   = 0, 0
        for i, (label_load, ulabel_loader) in enumerate(trainloader):
            #  Unpacking variables
            x_strong, x_weak, x_labels, x_policy = label_load
            u_strong, u_weak, u_labels, u_policy = ulabel_loader

            if scheduler.state == 2 and (end_warmup == False): # if exited warmup phaze
                ema.re_init(model)
                end_warmup = True

            if i % run_ctaugment == 0 and i > 0:
                model.eval()
                with torch.no_grad():
                    pred = model(x_strong.to(device, non_blocking=True)).softmax(1)
                    #mae = F.l1_loss(pred, torch.zeros(pred.size()).scatter_(1, x_labels.reshape(-1, 1), 1).to(device), reduction = 'none').sum(axis=1)
                    for y_pred, t, policy in zip(pred, x_labels, x_policy):
                        error = y_pred
                        error[t] -= 1
                        error = torch.abs(error).sum()
                        augmentation.update(policy, 1 - 0.5*error.item())
            
            model.train()
            with torch.no_grad():
                unlabeled_predictions    = model(u_weak.to(device, non_blocking=True))
                indexes, pseudolabels = lossfunc.get_pseudo(unlabeled_predictions)

            optimizer.zero_grad()
            strongly_augmented_selected = u_strong[indexes]
            # input_batch = torch.cat((x_weak, strongly_augmented_selected), 0)
            p_num = indexes.sum()
            # model_output = model(input_batch.to(device))
            # labeled_predictions, unlabeled_strong_predictions = torch.split(model_output, [x_weak.size()[0], p_num])

            labeled_predictions = model(x_weak.to(device, non_blocking=True))
            unlabeled_strong_predictions = model(u_strong.to(device, non_blocking=True))
            t_loss, s_loss, u_loss = lossfunc.calculate_losses(labeled_predictions, x_labels.to(device, non_blocking=True), unlabeled_strong_predictions[indexes], pseudolabels[indexes], indexes.size()[0])
            t_loss.backward()
            optimizer.step()
            scheduler.step()
            if end_warmup:
                ema.update()

            if i % update_bar == 0: 
                tb_writer.add_scalar('Loss/train', t_loss, itertrain)
                tb_writer.add_scalar('SupervisedLoss/train', s_loss, itertrain)
                tb_writer.add_scalar('UnsupervisedLoss/train', u_loss, itertrain)
                tb_writer.add_scalar('Psudolabel_num/train', p_num, itertrain)
                tb_writer.add_scalar('Learning_Rate/train', optimizer.param_groups[0]['lr'], itertrain)
                bar.update(update_bar)
            # CT augment update
                #network not mature for CTaugment
            

                    #augmentation.update(x_policy, 1 - 0.5*mae)

            # validation
            if i > 0 and i % run_validation == 0:
                correct, total  = 0, 0
                model.eval()
                with torch.no_grad(): #Turn off gradients
                    for X, Y in validation_loader:
                        y       = np.argmax(model(X.to(device, non_blocking=True)).cpu().numpy(), axis=1) == Y.numpy()
                        total  += len(y)
                        correct += np.sum(y)
                    tb_writer.add_scalar('Accuracy/validation', correct/total, iterval)

                    print('Validation on iteration k={0} yielded {1} accuracy'.format(i, correct/total))
                
                    if top_val < (correct/total):
                        save_models([model, 'normal'])
                        top_val = (correct/total)

            itertrain += 1
            iterval   += 1

def save_models(*models, prefix='current_run'):
    for model, name in models:
        save_dir = os.path.expanduser(args.save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, prefix + args.name_model_specs + "_" + args.use_database + '_' + name + '.state_dict'))


def collate_fn_weak(ims):
    global augmentation
    s = augmentation.weak_batch(ims)
    tensors, labels = [x[0] for x in s], [x[1] for x in s]

    return torch.stack(tensors), torch.LongTensor(labels)

def collate_fn_strong(ims, probe = False):
    global augmentation
    # print("previous_rates ", augmentation.cta.rates['autocontrast'][0])
    strong = augmentation.strong_batch(ims, probe)
    weak   = collate_fn_weak(ims)
    
    tensors, labels, policy = [x[0] for x in strong], [x[1] for x in strong], [x[2] for x in strong]
    s = torch.stack(tensors)

    #labels, policy = [x[0] for x in labels], labels[0][1]

    return s, weak[0], torch.LongTensor(labels), policy #, augmentation.cta.rates['autocontrast'][0]


def default_collate_fn(ims):
    
    tensors, labels = [default_transforms(x[0]) for x in ims], [x[1] for x in ims]

    s = torch.stack(tensors)

    return s, torch.LongTensor(labels)


if __name__ == "__main__":
    # from models.wideresnet import WideResNet
    from models.WideResNet import WideResNet
    import torch
    
    # tensorboard writer
    tb_writer = SummaryWriter(log_dir=args.tbsw_logdir)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K         = args.K                         # steps: 2**20 ideal
    B         = args.B                         # batch size: 64 ideal
    mu        = args.mu    
    strides = [1, 1, 2, 2]
    model = WideResNet(d=28, k=3, n_classes=10, input_features=3, output_features=16, strides=strides)                # prop unsup/sup: 7 ideal
    #model     = WideResNet(3, 28, 2, 10)

    
    # Creating Dataset
    labels_per_class  = [args.labels_per_class for _ in range(10)]
    dataset_loader    = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)
    dataset_loader.load()

    unlabeled_dataset                        = DataSet(dataset_loader.get_set(dataset_loader.unlabeled_set), batch=B*mu, steps=K)
    labeled_dataset                          = DataSet(dataset_loader.get_set(dataset_loader.labeled_set), batch=B, steps=K)
    unlabeled_dataset, validation_dataset    = unlabeled_dataset.split(p=0.9, seed=43)

    # Creating Data Loaders

    
    unlabeled_collate = functools.partial(collate_fn_strong, probe=False)
    labeled_collate   = functools.partial(collate_fn_strong, probe=True)

    v_loader      = DataLoader(validation_dataset, batch_size = (mu + 1)*B, collate_fn = default_collate_fn, num_workers=args.num_workers1, pin_memory = True, shuffle=True)
    lbl_loader    = DataLoader(labeled_dataset, batch_size = B, collate_fn = labeled_collate, num_workers = args.num_workers2, pin_memory = True, shuffle=True)
    ulbl_loader   = DataLoader(unlabeled_dataset, batch_size = mu*B, collate_fn = unlabeled_collate, num_workers = args.num_workers3, pin_memory = True, shuffle=True)


    #  Model Settings

    model.to(device)
    ema = EMA(model, decay = 0.999, device=device)
    save_models([model, "normal"], [ema.get(), "ema"])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = cosineLRreduce(optimizer, K, warmup=args.warmup_scheduler)

    train_fixmatch(model,ema, zip(lbl_loader, ulbl_loader), v_loader, augmentation, optimizer, scheduler, device, K, tb_writer)
    tb_writer.close()

    # Save everything
    save_models([model, 'normal'], [ema.get(), 'ema'], prefix='_final_' + str(datetime.datetime.now()))


