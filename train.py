
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
import numpy as np

parser = argparse.ArgumentParser(description='Dataset')

parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--root', type=str, default='/home/firedragon/databases/')
parser.add_argument('--save_dir', type=str, default='/home/firedragon/DeepLearningModels/')
parser.add_argument('--name_model_specs', type=str, default='FixMatchModel')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)

# parsed in main
parser.add_argument('--B', type = int, default = 20)     #64
parser.add_argument('--K', type = int, default = 100)  #220
parser.add_argument('--mu', type = int, default = 4)     # 7

args = parser.parse_args()


class fixmatch_Loss():

    def __init__(self, l=1, threshold=0.95):
        self.u_weight = l
        self.tau = threshold

    def __call__(self, labeled_prediction, labeled_labels,
                 unlabeled_weak_predictions = None, unlabeled_strong_predictions = None):
        
        indexes_over_threshold         = np.zeros(1)        # init for validation
        supervised                     = F.cross_entropy(labeled_prediction, labeled_labels)
        
        if unlabeled_weak_predictions != None:
            max_pred, pseudolabels     = unlabeled_weak_predictions.softmax(1).max(axis=1)
            indexes_over_threshold     = max_pred > self.tau
       
        if indexes_over_threshold.sum() > 0:
            over, total      = indexes_over_threshold.size()[0], indexes_over_threshold.sum()
            if unlabeled_strong_predictions != None:
                unsupervised = over / total * F.cross_entropy(unlabeled_strong_predictions[indexes_over_threshold], pseudolabels[indexes_over_threshold])
        else:
            return supervised
        return supervised + self.u_weight * unsupervised


def train_fixmatch(model, trainloader, augmentation, optimizer, scheduler, device, K, tb_writer, validationloader):
    lossfunc = fixmatch_Loss()
    with tqdm(total=K) as bar:
        iter    = 0
        iterval = 0

        for label_load, ulabel_load in trainloader:
            #  Unpacking variables
            x_strong, x_weak, x_labels, x_policy = label_load
            u_strong, u_weak, u_labels, u_policy = ulabel_load
            
            model.train()
            optimizer.zero_grad()
            
            x_weak              = x_weak.to(device)
            labeled_predictions = model(x_weak)  # weak
            
            with torch.no_grad():
                u_weak                = u_weak.to(device)
                unlabeled_predictions = model(u_weak)
            
            u_strong                     = u_strong.to(device)
            unlabeled_strong_predictions = model(u_strong)
            x_labels                     = x_labels.to(device)
            loss                         = lossfunc(labeled_predictions, x_labels, unlabeled_predictions, unlabeled_strong_predictions)
            print('train loss:', loss)

            tb_writer.add_scalar('Loss/train', loss, iter)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # CT augment update
            model.eval()
            with torch.no_grad():
                
                #  validation  #
                for get_val in validationloader:
                    _, img_val, label_val, _ = get_val
                    #print(img_val)
                    img_val                 = img_val.to(device)
                    label_val               = label_val.to(device)
                    validation_predictions  = model(img_val)
                    val_loss                = lossfunc(validation_predictions, label_val, None, None)
                    print('val loss:', val_loss)
                
                iterval += 1
                tb_writer.add_scalar('Loss/validation', val_loss, iterval)
                
                x_strong = x_strong.to(device)
                pred = model(x_strong).softmax(1)
                mae = F.l1_loss(pred, torch.zeros(pred.size()).to(device).scatter_(1, x_labels.reshape(-1, 1), 1))
                augmentation.update(x_policy, 1 - 0.5*mae)
            bar.update(1)
            iter += 1

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

    return s, weak[0], torch.LongTensor(labels), policy #, augmentation.cta.rates['autocontrast'][0]


if __name__ == "__main__":
    from models.wideresnet import WideResNet
    import torch
    
    # tensorboard writer
    tb_writer = SummaryWriter()
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K         = args.K                         # steps: 2**20 ideal
    B         = args.B                         # batch size: 64 ideal
    mu        = args.mu                        # prop unsup/sup: 7 ideal
    model     = WideResNet(3, 28, 2, 10)

    
    # Creating Dataset
    labels_per_class  = [10 for _ in range(10)]
    dataset_loader    = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)
    dataset_loader.load()

    unlabeled_dataset                        = DataSet(dataset_loader.get_set(dataset_loader.unlabeled_set), batch=B*mu, steps=K)
    labeled_dataset, labeled_dataset_val     = DataSet(dataset_loader.get_set(dataset_loader.labeled_set), batch=B, steps=K).split(p = 0.9)

    # Creating Data Loaders

    augmentation  = Augmentation()
    lblval_loader = DataLoader(labeled_dataset_val, batch_size = B, collate_fn = collate_fn_strong, num_workers = 1)
    lbl_loader    = DataLoader(labeled_dataset, batch_size = B, collate_fn = collate_fn_strong, num_workers = 1)
    ulbl_loader   = DataLoader(unlabeled_dataset, batch_size = mu*B, collate_fn = collate_fn_strong, num_workers = 3)

    #  Model Settings

    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = cosineLRreduce(optimizer, K)

    train_fixmatch(model,zip(lbl_loader, ulbl_loader), augmentation, optimizer, scheduler, device, K, tb_writer, lblval_loader)
    tb_writer.close()

    torch.save(model.state_dict(), args.save_dir + args.name_model_specs + '_' + args.use_database) # save trained weights and model arguments
