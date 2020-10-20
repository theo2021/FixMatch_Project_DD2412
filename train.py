# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
import argparse
# from torch import Tensor
# from torchvision.transforms import ToTensor
from scheduler.cosineLRreduce import cosineLRreduce
from torch.nn import functional as F
from data.data import DataSet, Augmentation, collate_fn_weak, collate_fn_strong, CustomLoader
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('--download', type=bool, default=True)
parser.add_argument('--root', type=str, default='/home/theo/databases/')
parser.add_argument('--use_database', type=str, default='CIFAR10')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--batchsize', type=int, default=2)
args = parser.parse_args()


class fixmatchloss():

    def __init__(self, l=1, threshold=0.95):
        self.u_weight = l
        self.tau = threshold

    def __call__(self, labeled_prediction, labeled_labels,
                 unlabeled_weak_predictions, unlabeled_strong_predictions):
        supervised = F.cross_entropy(labeled_prediction, labeled_labels)
        max_pred, pseudolabels = unlabeled_weak_predictions.softmax(1).max(axis=1)
        indexes_over_threshold = max_pred > self.tau
        if indexes_over_threshold.sum() > 0:
            over, total = indexes_over_threshold.size()[0], indexes_over_threshold.sum()
            unsupervised = over / total * F.cross_entropy(unlabeled_strong_predictions[indexes_over_threshold], pseudolabels[indexes_over_threshold])
        else:
            return supervised
        return supervised + self.u_weight * unsupervised


def train_fixmatch(model, trainloader, augmentation, optimizer, scheduler, device, K):
    lossfunc = fixmatchloss()
    with tqdm(total=K) as bar:
        for label_load, ulabel_load in trainloader:
            #  Unpacking variables
            labeled_samples, labeled_targets = label_load
            u_strong, u_weak, u_labels, policy = ulabel_load
            optimizer.zero_grad()
            labeled_predictions = model(labeled_samples)  # weak
            unlabeled_predictions = model(u_weak)
            unlabeled_strong_predictions = model(u_strong)
            loss = lossfunc(labeled_predictions, labels, unlabeled_predictions, unlabeled_strong_predictions)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # CT augment update
            
            bar.update(1)




if __name__ == "__main__":
    from models.wideresnet import WideResNet
    import torch

    K = 2**20
    B = 16
    mu = 7
    model = WideResNet(3, 28, 2, 10)

    # Creating Dataset
    labels_per_class  = [10 for _ in range(10)]
    dataset_loader = CustomLoader(labels_per_class = labels_per_class, db_dir = args.root, db_name = args.use_database, mode = args.task, download = args.download)
    dataset_loader.load()

    unlabeled_dataset = DataSet(dataset_loader.get_set(dataset_loader.unlabeled_set), batch=B*mu, steps=K)
    labeled_dataset   = DataSet(dataset_loader.get_set(dataset_loader.labeled_set), batch=B, steps=K)

    # Creating Data Loaders

    augmentation = Augmentation()
    lbl_loader  = DataLoader(labeled_dataset, batch_size = B, collate_fn = collate_fn_weak)
    ulbl_loader = DataLoader(unlabeled_dataset, batch_size = mu*B, collate_fn = collate_fn_strong, num_workers=3)

    #  Model Settings

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = cosineLRreduce(optimizer, K)

    train_fixmatch(model,zip(lbl_loader, ulbl_loader), augmentation, optimizer, scheduler, device, K)




    
    
    # j=0
    # for i in loader:
    #     print(ToTensor()(i[0][0]))
    #     if j > 2:
    #         exit()
    #     j+=1
    
    
    # Data = Augmentation(40)
    # loader = DataLoader(
    #     Data,batch_size=
    # )
    