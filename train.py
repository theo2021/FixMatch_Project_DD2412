import torch
from torch.nn import functional as F
from tqdm import tqdm
import os

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


def save_models(*models, saving_dir='', prefix='current_run'):
    for model, name in models:
        save_dir = os.path.expanduser(saving_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, prefix + '_' + name + '.state_dict'))


def train_fixmatch(train_loader, val_loader, model, K, augmentation, optimizer, scheduler, device, tb_writer, ema=None, saving_dir='', threshold=0.95):
    lossfunc = fixmatch_Loss(threshold=threshold)
    run_validation = 400
    update_bar = 20
    top_val = 0
    run_validation = 200
    warmup = True
    with tqdm(total = K) as bar:
        for i, loader in enumerate(train_loader):
            if i > K:
                break
            if scheduler.state == 2 and (ema is not None) and warmup:
                ema.register()
                warmup = False
                
            x_weak = loader['label_samples'].to(device, non_blocking=True)
            x_labels = loader['label_targets'].to(device, non_blocking=True)
            u_weak = loader['ulabel_samples_weak'].to(device, non_blocking=True)
            u_strong = loader['ulabel_samples_strong'].to(device, non_blocking=True)
            # pseudolabels
            model.train()
            with torch.no_grad():
                unlabeled_predictions = model(u_weak)
                indexes, pseudolabels = lossfunc.get_pseudo(unlabeled_predictions)
            
            optimizer.zero_grad()
            strongly_augmented_selected = u_strong[indexes]
            p_num = indexes.sum()
            labeled_predictions = model(x_weak)
            unlabeled_strong_predictions = model(u_strong)
            t_loss, s_loss, u_loss = lossfunc.calculate_losses(labeled_predictions, x_labels, unlabeled_strong_predictions[indexes], pseudolabels[indexes], indexes.size()[0])
            t_loss.backward()
            optimizer.step()
            scheduler.step()
            if warmup == False:
                ema.update()
            if i > 0 and i % update_bar == 0: 
                tb_writer.add_scalar('Loss/train', t_loss, i)
                tb_writer.add_scalar('SupervisedLoss/train', s_loss, i)
                tb_writer.add_scalar('UnsupervisedLoss/train', u_loss, i)
                tb_writer.add_scalar('Psudolabel_num/train', p_num, i)
                tb_writer.add_scalar('Learning_Rate/train', optimizer.param_groups[0]['lr'], i)
                bar.update(update_bar)

            #CTAugment
            if 'policies' in loader.keys():
                x_strong = loader['ct_samples'].to(device, non_blocking=True)
                x_policy = loader['policies']
                x_labels = loader['ct_labels'].to(device, non_blocking=True)
                model.eval()
                with torch.no_grad():
                    pred = model(x_strong).softmax(1)
                    #mae = F.l1_loss(pred, torch.zeros(pred.size()).scatter_(1, x_labels.reshape(-1, 1), 1).to(device), reduction = 'none').sum(axis=1)
                    for y_pred, t, policy in zip(pred, x_labels, x_policy):
                        error = y_pred
                        error[t] -= 1
                        error = torch.abs(error).sum()
                        augmentation.update_rates(policy, 1 - 0.5*error.item())
            
            if i % run_validation == 0 and i > 0:
                correct, total  = 0, 0
                model.eval()
                with torch.no_grad():
                    for X, Y in val_loader:
                        y = model(X.to(device, non_blocking=True)).max(axis=1)[1] == Y.to(device, non_blocking=True)
                        total += len(y)
                        correct += sum(y)
                acc = 1.0*correct/total
                tb_writer.add_scalar('Accuracy/validation', acc, i)
                print('Validation on iteration k={0} yielded {1} accuracy'.format(i, acc))
                if acc > top_val:
                    top_val = acc
                    save_models([model, 'normal'], saving_dir=saving_dir)

        save_models([model, 'normal'], saving_dir=saving_dir, prefix='final')
        if warmup == False:
            ema.apply_shadow()
            save_models([model, 'ema'], saving_dir=saving_dir, prefix='final')


            
                
            
