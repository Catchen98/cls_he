import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score,confusion_matrix
import cv2
from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW
from tqdm import tqdm
from albumentations import *
from albumentations.pytorch import ToTensor

from sklearn.model_selection import train_test_split
from dataset import LeafDataset,LeafPILDataset
import argparse
import os

from resnest import resnest101,resnest200
from res2net_v1b import res2net50_v1b, res2net101_v1b
from efficientnet_pytorch import EfficientNet
import pretrainedmodels as models
from logger import setup_logger


import warnings
warnings.filterwarnings("ignore")

# torch.set_num_threads(0)

def train_fn(net, loader, args,scheduler):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []

    pbar = tqdm(total=len(loader), desc='Training')

    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        # print(labels)
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
            # import pdb;pdb.set_trace()
            output = net(mixed_input)

            loss = l * _reduce_loss(loss_fn(output, target_a)) + (1 - l) * _reduce_loss(loss_fn(output, target_b))
        else:
            output = net(images)
            loss = loss_fn(output, labels)

        batch_size = images.size(0)
        (loss * batch_size).backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc, np.argmax(labels.cpu().detach().numpy(), 1)), 0)
        preds_for_acc = np.concatenate((preds_for_acc, np.argmax(output.cpu().detach().numpy(), 1)), 0)

        # labels_for_acc.append( labels.cpu().numpy())
        pbar.update()
    # labels_for_acc = np.vstack(labels_for_acc)
    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    

    pbar.close()
    print(confusion_matrix(labels_for_acc, preds_for_acc))
    return running_loss / TRAIN_SIZE, accuracy

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

def valid_fn(net, loader):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []

    pbar = tqdm(total=len(loader), desc='Validation')

    with torch.no_grad():  # torch.no_grad() prevents Autograd engine from storing intermediate values, saving memory
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            net.eval()
            predictions = net(images)
            loss = loss_fn(predictions, labels)

            loss = _reduce_loss(loss)

            running_loss += loss.item() * labels.shape[0]
            labels_for_acc = np.concatenate((labels_for_acc, np.argmax(labels.cpu().detach().numpy(), 1)), 0)
            preds_for_acc = np.concatenate((preds_for_acc, np.argmax(predictions.cpu().detach().numpy(), 1)), 0)
            # labels_for_acc.append( labels.cpu().numpy())
            pbar.update()
        
        # labels_for_acc = np.vstack(labels_for_acc)
        accuracy = accuracy_score(labels_for_acc, preds_for_acc)
        print(confusion_matrix(labels_for_acc, preds_for_acc))

    pbar.close()
    return running_loss / VALID_SIZE, accuracy

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


    arg('--dataset_type', type=str, default='pillow', help='choose from [pillow,cv2].')

    args = parser.parse_args()

    IMAGE_FOLDER = args.root + '/images/'
    os.makedirs(args.run_root, exist_ok=True)
    N_CLASSES = 4

    logger = setup_logger("plant pathology", args.run_root, 0)
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
    def get_image_path(filename):
        return (IMAGE_FOLDER + filename + '.jpg')


    train = pd.read_csv(args.root + '/train.csv')
    test = pd.read_csv(args.root + '/test.csv')
    if args.debug:
        logger.info('=> debug..')
        train = train.sample(n=80)

    train['image_path'] = train['image_id'].apply(get_image_path)
    test['image_path'] = test['image_id'].apply(get_image_path)
    train_labels = train.loc[:, 'healthy':'scab']
    train_paths = train.image_path
    test_paths = test.image_path

    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=0.2,
                                                                            random_state=23, stratify=train_labels)
    train_paths.reset_index(drop=True, inplace=True)
    train_labels.reset_index(drop=True, inplace=True)
    valid_paths.reset_index(drop=True, inplace=True)
    valid_labels.reset_index(drop=True, inplace=True)

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.n_epochs
    TRAIN_SIZE = train_labels.shape[0]
    VALID_SIZE = valid_labels.shape[0]
    MODEL_NAME = args.model
    device = 'cuda'
    lr = args.lr

    if args.dataset_type == 'pillow':
        train_dataset = LeafPILDataset(train_paths, train_labels, aug=args.aug,use_onehot=True)
    else:
        train_dataset = LeafDataset(train_paths, train_labels, aug=args.aug,use_onehot=True)

    trainloader = Data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    if args.dataset_type == 'pillow':
        valid_dataset = LeafPILDataset(valid_paths, valid_labels, train=False,use_onehot=True)
    else:
        valid_dataset = LeafDataset(valid_paths, valid_labels, train=False,use_onehot=True)

    validloader = Data.DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

    # model = EfficientNet.from_pretrained(MODEL_NAME)
    # num_ftrs = model._fc.in_features
    # model._fc = nn.Sequential(nn.Linear(num_ftrs,1000,bias=True),
    #                           nn.ReLU(),
    #                           nn.Dropout(p=0.5),
    #                           nn.Linear(1000,4, bias=True))

    if args.model in ['resnest50', 'resnest101', 'resnest200', 'resnest269']:
        model = eval(args.model)(model_path=args.model_path)
    elif args.model in ['cosine_resnest50', 'cosine_resnest101', 'cosine_resnest200', 'cosine_resnest269']:
        assert args.cosine_type is not None
        model = eval(args.model)(model_path=args.model_path, cosine_type=args.cosine_type, num_classes=N_CLASSES)
    elif args.model in ['res2net50_v1b', 'res2net101_v1b']:
        model = eval(args.model)(model_path=args.model_path, with_attention=args.with_attention)
    elif args.model in ['efficientnet-b5','efficientnet-b6','efficientnet-b7']:
        model = EfficientNet.from_pretrained(args.model)
    else:
        model = getattr(models, args.model)(pretrained='imagenet')
    feature_dim = model._fc.in_features if 'efficientnet' in args.model else model.last_linear.in_features
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
                # print (x.size())
                return F.avg_pool2d(x, x.shape[2:])
        model.avg_pool = AvgPool()
        model.avgpool = AvgPool()
        model.last_linear = nn.Linear(feature_dim, N_CLASSES)  # new add by dongb

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    num_train_steps = int(len(train_dataset) / BATCH_SIZE * NUM_EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataset) / BATCH_SIZE * 5,
                                                num_training_steps=num_train_steps)
    # import pdb;pdb.set_trace()
    if args.label_smoothing:
        # print('=> using label smoothing.')
        logger.info('=> using label smoothing.')
        loss_fn = CrossEntropyLabelSmooth(num_classes=N_CLASSES)
    else:
        # loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loss = []
    valid_loss = []
    train_acc = []
    val_acc = []

    best_acc = 0
    for epoch in range(NUM_EPOCHS):

        tl, ta = train_fn(model, loader=trainloader, args=args,scheduler=scheduler)
        vl, va = valid_fn(model, loader=validloader)
        train_loss.append(tl)
        valid_loss.append(vl)
        train_acc.append(ta)
        val_acc.append(va)

        if va > best_acc:
            torch.save(model.state_dict(), os.path.join(args.run_root, 'best-model.pt'))
            best_acc = va
        torch.save(model.state_dict(), os.path.join(args.run_root, 'model.pt'))
        _lr = scheduler.get_lr()[0]
        # printstr = 'Epoch: ' + str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(
        #     vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va) + ', Best acc: ' + str(best_acc)
        # import pdb;pdb.set_trace()
        printstr = 'Epoch: {:4f}'.format(epoch) + ', lr: {:4f}'.format(_lr)  + ', Train loss: {:4f}'.format(tl) + ', Val loss: {:4f}'.format(vl) + ', Train acc: {:4f}'.format(ta) + ', Val acc: {:4f}'.format(va) + ', Best acc: {:4f}'.format(best_acc)
        logger.info(printstr)
        tqdm.write(printstr)
