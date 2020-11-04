
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, matthews_corrcoef
import cv2
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from tqdm import tqdm
from albumentations import *
from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split,StratifiedKFold
from dataset import LeafDataset,LeafPILDataset
import argparse
import os

from resnet import resnet50
from resnest import resnest101,resnest200
from res2net_v1b import res2net50_v1b, res2net101_v1b
from efficientnet_pytorch import EfficientNet
import pretrainedmodels as models
from logger import setup_logger
from models.baseline import Baseline, Baseline_freeze
from models.cosine_baseline import CosineBaseline
from models.layers.cosine_loss import AdaCos,ArcFace,SphereFace,CosFace,ArcCos,CircleSoftmax, ArcSoftmax
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# torch.set_num_threads(0)

def train_fn(net, loader, args,scheduler, optimizer, loss_fn, TRAIN_SIZE):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []

    pbar = tqdm(total=len(loader), desc='Training')

    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        show_train_img_flag = False
        if show_train_img_flag:
            mean = np.expand_dims(np.asarray([0.485, 0.456, 0.406]), axis=(1,2))
            std = np.expand_dims(np.asarray([0.229, 0.224, 0.225]),axis=(1,2))
            images = images.cpu().numpy()
            # images = images*std + mean
            cv2.imwrite('vis_train.jpg',np.transpose((images[0]*std+mean)*255, (1,2,0))[:,:,::-1])
            images = torch.from_numpy(images).cuda()

        net.train()
        optimizer.zero_grad()
        # predictions = net(images)
        
        r = np.random.rand(1)
        if args.cutmix and r >= 0.5:
            # generate mixed sample
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # compute output
            output = net(images)
            loss = _reduce_loss(loss_fn(output, target_a)) * lam + _reduce_loss(loss_fn(output, target_b)) * (
                    1. - lam)
        elif args.mixup and r >= 0.5:
            l = np.random.beta(0.2, 0.2)
            idx = torch.randperm(images.size(0))
            input_a, input_b = images, images[idx]
            target_a, target_b = labels, labels[idx]

            mixed_input = l * input_a + (1 - l) * input_b

            output = net(mixed_input)

            loss = l * _reduce_loss(loss_fn(output, target_a)) + (1 - l) * _reduce_loss(loss_fn(output, target_b))
        elif args.specific_mixup and r >= 0.5:
            l = np.random.uniform(0.3, 0.7)
            idx = torch.randperm(images.size(0))
            input_a, input_b = images, images[idx]
            # target_a, target_b = labels, labels[idx]
            target_a = torch.zeros_like(labels)
            target_b = torch.zeros_like(labels)
            for idx_a,idx_b in enumerate(idx):
                target_a[idx_a] = labels[idx_a]
                target_b[idx_a] = labels[idx_b]
                if labels[idx_a] == 0:
                    target_a[idx_a] = target_b[idx_a] = labels[idx_b]
                elif labels[idx_b] == 0:
                    target_a[idx_a] = target_b[idx_a] = labels[idx_a]
                elif labels[idx_a] == 1 or labels[idx_b] == 1 or (labels[idx_a] == 2 and labels[idx_b] == 3) or (labels[idx_a] == 3 and labels[idx_b] == 2): # 2,3; 1,x;x,1
                    target_a[idx_a] = target_b[idx_a] = 1


            mixed_input = l * input_a + (1 - l) * input_b
            output = net(mixed_input)

            loss = l * _reduce_loss(loss_fn(output, target_a)) + (1 - l) * _reduce_loss(loss_fn(output, target_b))
        elif args.ricap and r>=0.5:
            image_h, image_w = images.shape[2:]
            beta = 0.3
            ratio = np.random.beta(beta, beta, size=2)
            w0, h0 = np.round(np.array([image_w, image_h]) * ratio).astype(np.int)
            w1, h1 = image_w - w0, image_h - h0
            ws = [w0, w1, w0, w1]
            hs = [h0, h0, h1, h1]

            patches = []
            target_list = []
            label_weights = []
            for w, h in zip(ws, hs):
                indices = torch.randperm(images.size(0))
                x0 = np.random.randint(0, image_w - w + 1)
                y0 = np.random.randint(0, image_h - h + 1)
                patches.append(images[indices, :, y0:y0 + h, x0:x0 + w])
                target_list.append(labels[indices])
                label_weights.append(h * w / (image_h * image_w))

            images = torch.cat(
                [torch.cat(patches[:2], dim=3),
                torch.cat(patches[2:], dim=3)], dim=2)
            output = model(images)
            
            loss = sum([
                weight * _reduce_loss(loss_fn(output, target))
                for target, weight in zip(target_list, label_weights)
            ])
        else:
            output = net(images)
            # output = net(images, labels)
            loss = loss_fn(output, labels)

        batch_size = images.size(0)
        (loss * batch_size).backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
        preds_for_acc = np.concatenate((preds_for_acc, np.argmax(output.cpu().detach().numpy(), 1)), 0)
        pbar.update()
    
    # optimizer.step()
    # scheduler.step()
    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    f1 = f1_score(labels_for_acc, preds_for_acc,average='macro')
    mcc = matthews_corrcoef(labels_for_acc, preds_for_acc)
    pbar.close()
    print(confusion_matrix(labels_for_acc, preds_for_acc))
    # return running_loss / TRAIN_SIZE, accuracy
    return running_loss / TRAIN_SIZE, accuracy, f1

def _reduce_loss(loss):
    # return loss.sum() / loss.shape[0]
    return loss

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def valid_fn(net, loader, loss_fn, VALID_SIZE):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []

    pbar = tqdm(total=len(loader), desc='Validation')

    with torch.no_grad():  # torch.no_grad() prevents Autograd engine from storing intermediate values, saving memory
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            net.eval()
            if not args.arc_bnbias_flag:
                predictions = net(images)
            else:
                predictions = net(images, ret_score=True)
            
            # import ipdb;ipdb.set_trace()
            loss = loss_fn(predictions, labels)

            loss = _reduce_loss(loss)

            running_loss += loss.item() * labels.shape[0]
            labels_for_acc = np.concatenate((labels_for_acc, labels.cpu().numpy()), 0)
            preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
            pbar.update()

        accuracy = accuracy_score(labels_for_acc, preds_for_acc)
        f1 = f1_score(labels_for_acc, preds_for_acc,average='macro')
        print(confusion_matrix(labels_for_acc, preds_for_acc))

    pbar.close()
    return running_loss / VALID_SIZE, accuracy, f1


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'mean', ) -> torch.Tensor:
    r"""Function that computes Focal loss.
    See :class:`fastreid.modeling.losses.FocalLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {}".format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot = F.one_hot(target, num_classes=input.shape[1])

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


class FocalLoss(object):
    r"""Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = FocalLoss(cfg)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    # def __init__(self, alpha: float, gamma: float = 2.0,
    #              reduction: str = 'none') -> None:
    def __init__(self, alpha, gamma):
        self._alpha: float = alpha
        self._gamma: float = gamma
        
    def __call__(self, pred_class_logits: torch.Tensor, gt_classes: torch.Tensor) :
        loss = focal_loss(pred_class_logits, gt_classes, self._alpha, self._gamma)
        return loss 


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # if len(log_probs.size()) == 2:
        #     targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        # else:
        #     targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).unsqueeze(2).repeat(1,1,log_probs.size()[-1]).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init_kaiming(m):
    # import ipdb;ipdb.set_trace()
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def train(train_paths, train_labels, valid_paths, valid_labels, fold_idx=None):
    train_paths.reset_index(drop=True, inplace=True)  #将行号从乱序变为顺序
    train_labels.reset_index(drop=True, inplace=True)
    valid_paths.reset_index(drop=True, inplace=True)
    valid_labels.reset_index(drop=True, inplace=True)

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.n_epochs
    TRAIN_SIZE = train_labels.shape[0]
    VALID_SIZE = valid_labels.shape[0]
    MODEL_NAME = args.model
    lr = args.lr
    # args.arc_bnbias_flag = False
    
    if args.dataset_type == 'pillow':
        print('=> using pillow..')
        train_dataset = LeafPILDataset(train_paths, train_labels, aug=args.aug)
    else:
        train_dataset = LeafDataset(train_paths, train_labels, aug=args.aug)
    
    if args.resample_flag:
        weights = []
        labels = []
        try:
            for data,label in train_dataset:
                labels.append(label.item())
        except:
            pass
        dict_label = Counter(labels)
        try:
            # weights = [1 / dict_label[label.item()] for data, label in train_dataset]
            for data,label in train_dataset:
                weights.append(1 / dict_label[label.item()])
                # import ipdb;ipdb.set_trace()
                # if label == 0:
                #     weights.append(20)
                # elif any(label == _ for _ in [5,9,10,15]):
                #     weights.append(2)
                # else:
                #     weights.append(1)
        except:
            pass
        sampler = WeightedRandomSampler(weights,len(weights),replacement=True)
    else:
        trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    if args.dataset_type == 'pillow':
        print('=> using pillow..')
        valid_dataset = LeafPILDataset(valid_paths, valid_labels, train=False)
    else:
        valid_dataset = LeafDataset(valid_paths, valid_labels, train=False)

    validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

   
    if args.model in ['resnest50', 'resnest101', 'resnest200', 'resnest269']:
        model = eval(args.model)(model_path=args.model_path)
    elif args.model in ['cosine_resnest50', 'cosine_resnest101', 'cosine_resnest200', 'cosine_resnest269']:
        assert args.cosine_type is not None
        model = eval(args.model)(model_path=args.model_path, cosine_type=args.cosine_type, num_classes=N_CLASSES)
    elif args.model in ['res2net50_v1b', 'res2net101_v1b']:
        model = eval(args.model)(model_path=args.model_path, with_attention=args.with_attention)
    elif args.model in ['efficientnet-b5','efficientnet-b6','efficientnet-b7']:
        model = EfficientNet.from_pretrained(args.model)
    elif args.arc_bnbias_flag and (args.model in ['resnet50']):
        model = Baseline(num_classes=N_CLASSES, last_stride=2, model_path = '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnet50-19c8e357.pth',  use_bnneck=True)
        # model = CosineBaseline(num_classes=N_CLASSES, last_stride=2, model_path = '/data/hejy/hejy_dp/model_checkpoint_pytorch/resnet50-19c8e357.pth',cosine_loss_type = 'ArcFace')#, m=0.5)
    else:
        model = getattr(models, args.model)(pretrained='imagenet')
        # model = getattr(models, args.model)(pretrained=None)
    if hasattr(model,'classifier'):
        if hasattr(model.classifier, 'in_features'):
            feature_dim = model.classifier.in_features
        else:
            feature_dim = model.classifier.num_features
    else:
        feature_dim = model._fc.in_features if 'efficientnet' in args.model else model.last_linear.in_features
    if not args.arc_bnbias_flag:
        if 'efficientnet' in args.model:
            # print('=> using efficientnet.')
            logger.info('=> using efficientnet.')
            model._fc = nn.Linear(feature_dim, N_CLASSES)  # new add by dongb
            # model._fc = nn.Sequential(nn.Linear(feature_dim,1000,bias=True),
            #                   nn.ReLU(),
            #                   nn.Dropout(p=0.5),
            #                   nn.Linear(1000,N_CLASSES, bias=True))
        else:
            class AvgPool(nn.Module):
                def forward(self, x):
                    return F.avg_pool2d(x, x.shape[2:])
            model.avg_pool = AvgPool()
            model.avgpool = AvgPool()
            model.last_linear = nn.Linear(feature_dim, N_CLASSES)  # new add by dongb
   
    if args.model_path is not None:
        state = torch.load(str(args.model_path))
        model.load_state_dict(state)
        if args.freeze_backbone_flag:
            for param in model.base.parameters():
                    param.requires_grad = False
            model.bottleneck.apply(weights_init_kaiming)
            model.classifier.apply(weights_init_classifier)
    model.to(device)
    

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_train_steps = int(len(train_dataset) / BATCH_SIZE * NUM_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset) / BATCH_SIZE * 5,
                                                num_training_steps=num_train_steps)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,60], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2, gamma=0.1)
    if args.label_smoothing:
        # print('=> using label smoothing.')
        logger.info('=> using label smoothing.')
        loss_fn = CrossEntropyLabelSmooth(num_classes=N_CLASSES)
    elif args.focal_loss:
        loss_fn = FocalLoss(alpha=0.25, gamma=2)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    train_loss = []
    valid_loss = []
    train_acc = []
    val_acc = []

    best_acc = 0
    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        tl, ta, tf = train_fn(model, loader=trainloader, args=args, scheduler=scheduler, optimizer=optimizer, loss_fn=loss_fn, TRAIN_SIZE=TRAIN_SIZE)
        vl, va, vf = valid_fn(model, loader=validloader, loss_fn=loss_fn, VALID_SIZE=VALID_SIZE)
        train_loss.append(tl)
        valid_loss.append(vl)
        train_acc.append(ta)
        val_acc.append(va)
        
        if fold_idx is None:
            if vf > best_f1:
                torch.save(model.state_dict(), os.path.join(args.run_root, 'best-model.pt'))
                best_f1 = vf
            if va > best_acc:
                best_acc = va
            torch.save(model.state_dict(), os.path.join(args.run_root, 'model.pt'))
            _lr = scheduler.get_lr()[0]
            # _lr = scheduler.get_last_lr()[0]  #adapt for torch.optim.lr_scheduler
            # printstr = 'Epoch: ' + str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(
            #     vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va) + ', Best acc: ' + str(best_f1)
            # import pdb;pdb.set_trace()
            # printstr = 'Epoch: {:4f}'.format(epoch) + ', lr: {:4f}'.format(_lr)  + ', Train loss: {:4f}'.format(tl) + ', Val loss: {:4f}'.format(vl) + ', Train acc: {:4f}'.format(ta) + ', Val acc: {:4f}'.format(va) + ', Best acc: {:4f}'.format(best_f1)
            printstr = 'Epoch: {:4f}'.format(epoch) + ', lr: {:4f}'.format(_lr)  + ', Train loss: {:4f}'.format(tl) + ', Val loss: {:4f}'.format(vl) + ',Train acc: {:4f}'.format(ta) + ', Val acc: {:4f}'.format(va) + ', Train f1: {:4f}'.format(tf) + ', Val f1: {:4f}'.format(vf) + ', Best acc: {:4f}'.format(best_acc) + ', Best f1: {:4f}'.format(best_f1)
        else:
            if vf > best_f1:
                torch.save(model.state_dict(), os.path.join(args.run_root, 'fold{}_best-model.pt'.format(fold_idx)))
                best_f1 = vf
            if va > best_acc:
                best_acc = va
            torch.save(model.state_dict(), os.path.join(args.run_root, 'fold{}_model.pt'.format(fold_idx)))
            _lr = scheduler.get_lr()[0]
            printstr = 'Fold_{} '.format(fold_idx) + 'Epoch: {:4f}'.format(epoch) + ', lr: {:4f}'.format(_lr)  + ', Train loss: {:4f}'.format(tl) + ', Val loss: {:4f}'.format(vl) + ',Train acc: {:4f}'.format(ta) + ', Val acc: {:4f}'.format(va) + ', Train f1: {:4f}'.format(tf) + ', Val f1: {:4f}'.format(vf) + ', Best acc: {:4f}'.format(best_acc) + ', Best f1: {:4f}'.format(best_f1)
        # optimizer.step()
        # scheduler.step()   
            
        logger.info(printstr)
        tqdm.write(printstr)

if __name__ == '__main__':

    # hyper parameters
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('run_root')
    arg('--model', default='efficientnet-b5')
    arg('--batch-size', type=int, default=8)
    arg('--workers', type=int, default=4)
    arg('--lr', type=float, default=8e-4)
    arg('--n-epochs', type=int, default=30)
    arg('--tta', type=int, default=4)

    arg('--model_path', type=str, default=None)
    arg('--with_attention', type=bool, default=False)

    arg('--aug', type=str, default=None)
    arg('--pool_type', type=str, default=None)
    arg('--root', type=str, default='')
    arg('--cutmix', type=bool, default=False)
    arg('--mixup', type=bool, default=False)
    arg('--focal_loss', type=bool, default=False)
    arg('--label_smoothing', type=bool, default=False)
    arg('--cosine_type', type=str, default=None)
    
    arg('--debug', action='store_true', help='use debug')
    arg('--specific_mixup', action='store_true', help='use specific_mixup')
    arg('--ricap', action='store_true', help='use ricap')

    arg('--dataset_type', type=str, default='pillow', help='choose from [pillow,cv2].')
    arg('--arc_bnbias_flag', type=bool, default=False)
    arg('--freeze_backbone_flag', type=bool, default=False)
    arg('--resample_flag', type=bool, default=False )

    args = parser.parse_args()

    os.makedirs(args.run_root, exist_ok=True)
    N_CLASSES = 15
    device = 'cuda'
    logger = setup_logger("Chart", args.run_root, 0)
    logger.info(args)

    if args.cutmix:
        # print('=> using cutmix.')
        logger.info('=> using cutmix.')

    if args.mixup:
        # print('=> using mixup.')
        logger.info('=> using mixup.')

    if args.specific_mixup:
        # print('=> using specific_mixup.')
        logger.info('=> using specific_mixup.')
    
    if args.ricap:
        logger.info('=> using ricap.')
    
    train_data = pd.read_csv(args.root + '/PMC_2020_split_train.csv')
    test_data = pd.read_csv(args.root + '/PMC_2020_split_val.csv')

    if args.debug:
        logger.info('=> debug..')
        train = train.sample(n=120)
   
    train_labels = train_data.loc[:, 'area':'vertical_interval']
    test_labels = test_data.loc[:, 'area':'vertical_interval']

    train_image_id = train_data['image_id'].tolist()
    
    
    def get_image_path(filename):
        return filename
    train_data['image_path'] = train_data['image_id'].apply(get_image_path)
    test_data['image_path'] = test_data['image_id'].apply(get_image_path)
    
    train_paths = train_data.image_path
    test_paths = test_data.image_path
    kflod_flag = False
    n_splits = 5
    if not kflod_flag:
        # train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=0.2,
        #                                                                         random_state=23, stratify=train_labels)
        # train(train_paths, train_labels, valid_paths, valid_labels)
        train(train_paths, train_labels, test_paths, test_labels)
    else:
        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        # paths_placeholder = np.zeros(train_paths.values.shape[0])
        paths_placeholder = train_paths.values
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(paths_placeholder, train_labels.values.argmax(1))):
            _train_paths, _valid_paths = train_paths[train_idx], train_paths[valid_idx]
            _train_labels, _valid_labels = train_labels.loc[train_idx], train_labels.loc[valid_idx]
            train(_train_paths, _train_labels, _valid_paths, _valid_labels, fold_idx)

