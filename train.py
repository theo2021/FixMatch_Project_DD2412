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
parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--tbsw_logdir', type=str, default=None)
parser.add_argument('--root', type=str, default='~/databases/')              #sudo mkdir /home/databases and then sudo chmod -R 777 /home/databases (for permissions) 
parser.add_argument('--save_dir', type=str, default='~/DeepLearningModels/')
parser.add_argument('--name_model_specs', type=str, default='FixMatchModel')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)

# parsed in main
parser.add_argument('--B', type = int, default = 16)     # 64
parser.add_argument('--K', type = int, default = 1)    # 220
parser.add_argument('--mu', type = int, default = 7)     # 7

args = parser.parse_args()

augmentation  = Augmentation()


class fixmatch_Loss():

    def __init__(self, l=1, threshold=0.95):
        self.u_weight = l
        self.tau = threshold

    def __call__(self, labeled_prediction, labeled_labels,
                 unlabeled_weak_predictions, unlabeled_strong_predictions):
        supervised = F.cross_entropy(labeled_prediction, labeled_labels)
        max_pred, pseudolabels = unlabeled_weak_predictions.softmax(1).max(axis=1)
        indexes_over_threshold = max_pred > self.tau
        if indexes_over_threshold.sum() > 0:
            total, over = indexes_over_threshold.size()[0], indexes_over_threshold.sum()
            unsupervised = (over / total) * F.cross_entropy(unlabeled_strong_predictions[indexes_over_threshold], pseudolabels[indexes_over_threshold])
        else:
            return supervised
        return supervised + self.u_weight * unsupervised


def train_fixmatch(model, ema, trainloader, validation_loader, augmentation, optimizer, scheduler, device, K, tb_writer):
    # if model is confident for this threshold start the unlabeled and ctaugment
    confidence_threshold = 35
    confidence_sum = 0
    lossfunc = fixmatch_Loss()
    run_validation  = 200
    with tqdm(total = K) as bar:
        itertrain, iterval   = 0, 0
        for i, (label_load, ulabel_loader) in enumerate(trainloader):
            #  Unpacking variables
            x_strong, x_weak, x_labels, x_policy = label_load
            model.train()
            optimizer.zero_grad()
            labeled_predictions = model(x_weak.to(device))  # weak
            if confidence_sum < confidence_threshold:
                # no need to train the whole network at start since network isn't even confident for the training predictions
                confidence_sum += (labeled_predictions.softmax(1).max(axis=1)[0] > 0.95).sum()
                loss = F.cross_entropy(labeled_predictions.to(device), x_labels.to(device))
            else:
                u_strong, u_weak, u_labels, u_policy = ulabel_loader          
                with torch.no_grad():
                    unlabeled_predictions    = model(u_weak.to(device))
                unlabeled_strong_predictions = model(u_strong.to(device))
                loss = lossfunc(labeled_predictions, x_labels.to(device), unlabeled_predictions, unlabeled_strong_predictions)
            print('train loss:', loss)
            print('over_confidence', confidence_sum, 'lr', optimizer.param_groups[0]['lr'])

            tb_writer.add_scalar('Loss/train', loss, itertrain)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()

            # CT augment update
            if confidence_sum > confidence_threshold:
                #network not mature for CTaugment
                model.eval()
                with torch.no_grad():
                    pred = model(x_strong.to(device)).softmax(1)
                    mae = F.l1_loss(pred, torch.zeros(pred.size()).scatter_(1, x_labels.reshape(-1, 1), 1).to(device))
                    augmentation.update(x_policy, 1 - 0.5*mae)
            bar.update(1)

            # validation
            if i > 0 and i % run_validation == 0:
                correct, total  = 0, 0
                model.eval()
                with torch.no_grad(): #Turn off gradients
                    for X, Y in validation_loader:
                        y       = np.argmax(model(X.to(device)).cpu().numpy(), axis=1) == Y.numpy()
                        total  += len(y)
                        correct += np.sum(y)
                    tb_writer.add_scalar('Accuracy/validation', correct/total, iterval)

                    print('Validation on iteration k={0} yielded {1} accuracy'.format(i, correct/total))

            itertrain += 1
            iterval   += 1

                

def collate_fn_weak(ims):
    global augmentation
    s = augmentation.weak_batch(ims)
    tensors, labels = [x[0] for x in s], [x[1] for x in s]

    return torch.stack(tensors), torch.LongTensor(labels)

def collate_fn_strong(ims):
    global augmentation
    # print("previous_rates ", augmentation.cta.rates['autocontrast'][0])
    strong = augmentation.strong_batch(ims)
    weak   = collate_fn_weak(ims)
    
    tensors, labels = [torch.tensor(x[0]) for x in strong], [x[1] for x in strong]
    s = torch.stack(tensors)

    labels, policy = [x[0] for x in labels], labels[0][1]

    return s, weak[0], torch.LongTensor(labels), policy #, augmentation.cta.rates['autocontrast'][0]


def default_collate_fn(ims):
    
    tensors, labels = [torch.FloatTensor(np.array(x[0]).transpose(2, 0, 1))/255 for x in ims], [x[1] for x in ims]

    s = torch.stack(tensors)

    return s, torch.LongTensor(labels)


if __name__ == "__main__":
    from models.wideresnet import WideResNet
    # from models.WideResNet import WideResNet
    import torch
    
    # tensorboard writer
    tb_writer = SummaryWriter(log_dir=args.tbsw_logdir)
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K         = args.K                         # steps: 2**20 ideal
    B         = args.B                         # batch size: 64 ideal
    mu        = args.mu    
    strides = [1, 1, 2, 2]
    # model = WideResNet(d=28, k=3, n_classes=10, input_features=3, output_features=16, strides=strides)                # prop unsup/sup: 7 ideal
    model     = WideResNet(3, 28, 2, 10)

    
    # Creating Dataset
    labels_per_class  = [100 for _ in range(10)]
    dataset_loader    = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)
    dataset_loader.load()

    unlabeled_dataset                        = DataSet(dataset_loader.get_set(dataset_loader.unlabeled_set), batch=B*mu, steps=K)
    labeled_dataset                          = DataSet(dataset_loader.get_set(dataset_loader.labeled_set), batch=B, steps=K)
    unlabeled_dataset, validation_dataset    = unlabeled_dataset.split(p=0.9, seed=43)

    # Creating Data Loaders

    
    v_loader    = DataLoader(validation_dataset, batch_size = (mu + 1)*B, collate_fn = default_collate_fn, num_workers=1, pin_memory = True, shuffle=True)
    lbl_loader    = DataLoader(labeled_dataset, batch_size = B, collate_fn = collate_fn_strong, num_workers = 1, pin_memory = True, shuffle=True)
    ulbl_loader   = DataLoader(unlabeled_dataset, batch_size = mu*B, collate_fn = collate_fn_strong, num_workers = 3, pin_memory = True, shuffle=True)

    #  Model Settings

    model.to(device)
    ema = EMA(model, decay = 0.999)
    ema.register()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.0005, nesterov=True) # lr should be 0.03
    scheduler = cosineLRreduce(optimizer, K)

    train_fixmatch(model,ema, zip(lbl_loader, ulbl_loader), v_loader, augmentation, optimizer, scheduler, device, K, tb_writer)
    tb_writer.close()


    # Save everything

    save_dir, prefix = os.path.expanduser(args.save_dir), str(datetime.datetime.now())
    
    torch.save(model.state_dict(), os.path.join(save_dir, prefix + args.name_model_specs + "_" + args.use_database + '.state_dict'))
    ema.apply_shadow()
    torch.save(model.state_dict(), os.path.join(save_dir, prefix + args.name_model_specs + "_" + args.use_database + '_EMA' '.state_dict'))

    
    
