# import torch
# from Models.wideresnet import WideResNet
from tqdm import tqdm
from torch import Tensor
from torchvision.transforms import ToTensor
from scheduler.cosineLRreduce import cosineLRreduce
from torch.nn import functional as F
from data.data import Augmentation, DataLoader, DataSet
# from torch.utils.data import DataLoader



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
        for unlabeled_batch, labeled_batch, labels in trainloader:
            #unlabeled_batch = augmentation.weak(unlabeled_batch)
            # unlabeled_batch, labeled_batch, labels = [out.to(device) for out in trainloader]
            optimizer.zero_grad()
            labeled_predictions = model(augmentation(labeled_batch)) # weak
            print("OK")
            unlabeled_predictions = model(unlabeled_batch)
            print("OK2")
            print(labels)
            unlabeled_strong_predictions = model(augmentation(unlabeled_batch))
            loss = lossfunc(labeled_predictions, labels, unlabeled_predictions, unlabeled_strong_predictions)
            loss.backward()
            optimizer.step()
            scheduler.step()
            bar.update(1)




if __name__ == "__main__":
    from models.wideresnet import WideResNet
    import torch

    model = WideResNet(3, 28, 2, 10)
    K = 2**20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = cosineLRreduce(optimizer, K)

    labels_per_class  = [10 for _ in range(10)]
    dataset = DataSet(labels_per_class = labels_per_class, db_dir="/home/theo/databases/")
    dataset.load()

    loader = DataLoader(dataset, B= 4, mu=7, num_classes=10, K=K)
    def aug(im):
        return im
    
    # j=0
    # for i in loader:
    #     print(ToTensor()(i[0][0]))
    #     if j > 2:
    #         exit()
    #     j+=1
    train_fixmatch(model,loader, aug, optimizer, scheduler, device, K)
    
    # Data = Augmentation(40)
    # loader = DataLoader(
    #     Data,batch_size=
    # )
    