import torch
import torch.utils.data as Data
import torch.nn as nn
from torchvision import transforms
import numpy as np
from numpy import random
import pandas as pd
from PIL import Image
from albumentations import *
# from albumentations.pytorch import ToTensor
import os.path as osp
import cv2

import torchvision.transforms as T
from autoaugment import ImageNetPolicy,albuImageNetPolicy

import albumentations as albu
from transforms import RandomErasing



cv2.setNumThreads(4)

class LeafDataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, train=True, test=False, aug=None, use_onehot=False):
        self.paths = image_paths
        self.test = test
        self.use_onehot = use_onehot
        if self.test == False:
            self.labels = labels
        self.train = train
        # self.transform = albu.Compose([albu.HorizontalFlip(p=0.5),
        #                           albu.VerticalFlip(p=0.5),
        #                           albu.ShiftScaleRotate(rotate_limit=25.0, p=0.7),
        #                           albu.OneOf([albu.IAAEmboss(p=1),
        #                                  albu.IAASharpen(p=1),
        #                                  albu.Blur(p=1)], p=0.5),
        #                           albu.IAAPiecewiseAffine(p=0.5),
        #                           albu.Resize(545, 545, always_apply=True),
        #                           albu.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #                           albu.pytorch.ToTensor(),
        #                           ])

        self.transform = []
        self.transform.append(albu.HorizontalFlip(p=0.5))
        self.transform.append(albu.VerticalFlip(p=0.5))
        if aug is not None and 'rt90' in aug:
            print('=> using rotate 90.')
            self.transform.append(albu.RandomRotate90(p=0.5))
        self.transform.append(albu.ShiftScaleRotate(rotate_limit=25.0, p=0.7))
        self.transform.append(albu.OneOf([albu.IAAEmboss(p=1),
                                         albu.IAASharpen(p=1),
                                         albu.Blur(p=1)], p=0.5))
        self.transform.append(albu.IAAPiecewiseAffine(p=0.5))
        if aug is not None and 'cj' in aug:
            print('=> using color jittering.')
            self.transform.append(albu.RandomBrightness(p=0.5))
            self.transform.append(albu.HueSaturationValue(p=0.5))
        
        self.transform.append(albu.Resize(545, 545, always_apply=True))
        # self.transform.append(albu.Resize(224, 224, always_apply=True))
        if aug is not None and 'autoaug' in aug:
            print('=> using autoaug.')
            self.transform.append(albuImageNetPolicy())
        self.transform.append(albu.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        self.transform.append(albu.pytorch.ToTensor())
        self.transform = albu.Compose(self.transform)
        self.usere = False                  
        if aug is not None and 're' in aug:
            self.usere = True
            print('=> using random erasing.')
            self.re = RandomErasing()
        self.default_transform = albu.Compose([albu.Resize(545, 545, always_apply=True),
        # self.default_transform = albu.Compose([albu.Resize(224, 224, always_apply=True),
                                          albu.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), always_apply=True),
                                          albu.pytorch.ToTensor()])  # normalized for pretrained network

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, i):
        image = self.load_image(self.paths[i])
        if self.test == False and self.use_onehot == False:
            label = torch.tensor(np.argmax(self.labels.loc[i,
                                           :].values))  # loss function used later doesnt take one-hot encoded labels, so convert it using argmax
        elif self.test == False and self.use_onehot == True:
            label = torch.tensor(self.labels.loc[i,:].values.astype(np.float))
        if self.train:
            image = self.transform(image=image)['image']
            if self.usere:
                image = self.re(image)
        else:
            # print(">>>>>>>>>>>>>>>")
            image = self.default_transform(image=image)['image']

        if self.test == False:
            return image, label
        return image

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class Rank1Aug(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.zhiguang = True
        if self.zhiguang:
            self.blur = albu.Blur(blur_limit=3,p=0.5)
            self.chanelshuffle = albu.ChannelShuffle(p=0.5)
            self.hueSaturation = albu.HueSaturationValue(p=0.5)
            #albu.InvertImg(p=0.2),
            self.togray = albu.ToGray(p=0.2)
        else:
            self.p = p
            self.scr = albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=15, p=self.p)
            self.ig = albu.IAAAdditiveGaussianNoise(p=self.p)
            self.ipa = albu.IAAPiecewiseAffine(p=self.p)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if self.zhiguang:
            img = self.blur(image=img)['image']
            img = self.chanelshuffle(image=img)['image']
            img = self.hueSaturation(image=img)['image']
            img = self.togray(image=img)['image']
        else:
            img = self.scr(image=img)['image']
            img = self.ig(image=img)['image']
            img = self.ipa(image=img)['image']
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class LeafPILDataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, train=True, test=False, aug=None, use_onehot=False):
        self.paths = image_paths
        self.test = test
        self.use_onehot = use_onehot

        if self.test == False:
            self.labels = labels
        self.train = train

        self.transform = []
        # self.transform.append(T.Resize((224,224),interpolation=Image.LANCZOS))
        self.transform.append(T.Resize((448,448),interpolation=Image.LANCZOS))

        self.transform.append(T.RandomHorizontalFlip())
        # self.transform.append(T.RandomVerticalFlip())
        # self.transform.append(T.RandomCrop((192,192)))
        self.transform.append(T.RandomCrop((384,384)))
        if aug is not None and 'autoaug' in aug:
            print('=> using auto augmentation.')
            self.transform.append(ImageNetPolicy(fillcolor=(128, 128, 128)))
        if aug is not None and 'albu' in aug:
            print('=> using albu augmentation.')
            self.transform.append(Rank1Aug(p=0.5))
        if aug is not None and 'cj' in aug:
            print('=> using color jittering.')
            self.transform.append(T.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01))

        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform.append(RandomErasing())
        self.transform = T.Compose(self.transform)

        self.default_transform = []
        # # self.default_transform.append(T.RandomCrop((832,832)))
        # self.default_transform.append(T.Resize((224,224),interpolation=Image.LANCZOS))
        # self.default_transform.append(T.CenterCrop((192,192)))
        self.default_transform.append(T.Resize((448,448),interpolation=Image.LANCZOS))
        self.default_transform.append(T.CenterCrop((384,384)))
        self.default_transform.append(T.ToTensor())
        self.default_transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.default_transform = T.Compose(self.default_transform)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, i):
        # import ipdb;ipdb.set_trace()
        image = self.load_image(self.paths[i])
        
        # if self.test == False:
        #     label = torch.tensor(np.argmax(self.labels.loc[i,
        #                                    :].values))  # loss function used later doesnt take one-hot encoded labels, so convert it using argmax
        if self.test == False and self.use_onehot == False:
            label = torch.tensor(np.argmax(self.labels.loc[i,
                                           :].values))  # loss function used later doesnt take one-hot encoded labels, so convert it using argmax
        elif self.test == False and self.use_onehot == True:
            label = torch.tensor(self.labels.loc[i,:].values.astype(np.float)) 
        if self.train:
            image = self.transform(image)
        else:
            image = self.default_transform(image)
        
        
        if self.test == False:
            return image, label
            # return image, label, self.paths[i]
        return image

    def load_image(self, path):
        image = cv2.imread(path)
        # if self.train:
        #     image =self.expand(image)
        # print(path,'\n', image)
        # import ipdb;ipdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    def expand(self, image):
        scale = random.uniform(0.8,1.2)
        angle = 0.
        width, height = image.shape[1], image.shape[0]
    
        cx = 0.5 * width
        cy = 0.5 * height
        
        # if randrf(0, 1) > 0.5:
        #     angle = randrf(-30, 30)
    
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        # M[0, 2] -= cx - width * 0.5
        # M[1, 2] -= cy - height * 0.5      
        image = cv2.warpAffine(image, M, (width, height))
    
        return image

class LeafPILDataset_4pred(Data.Dataset):
    def __init__(self, image_paths, labels=None, train=True, test=False, aug=None, use_onehot=False,use_tencrop=False):
        self.paths = image_paths
        self.test = test
        self.use_onehot = use_onehot
        self.use_tencrop = use_tencrop

        if self.test == False:
            self.labels = labels
        self.train = train

        self.transform = []
        self.transform.append(T.Resize((224,224),interpolation=Image.LANCZOS))
        self.transform.append(T.RandomHorizontalFlip())
        # self.transform.append(T.RandomVerticalFlip())
        self.transform.append(T.RandomCrop((192,192)))
        if aug is not None and 'autoaug' in aug:
            print('=> using auto augmentation.')
            self.transform.append(ImageNetPolicy(fillcolor=(128, 128, 128)))
        if aug is not None and 'albu' in aug:
            print('=> using albu augmentation.')
            self.transform.append(Rank1Aug(p=0.5))
        if aug is not None and 'cj' in aug:
            print('=> using color jittering.')
            self.transform.append(T.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.01))

        self.transform.append(T.ToTensor())
        self.transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform.append(RandomErasing())
        self.transform = T.Compose(self.transform)

        self.default_transform = []
        # self.default_transform.append(T.RandomCrop((832,832)))
        # self.default_transform.append(T.Resize((224,224),interpolation=Image.LANCZOS))
        self.default_transform.append(T.Resize((448,448),interpolation=Image.LANCZOS))
        # self.default_transform.append(T.Resize((672,672),interpolation=Image.LANCZOS))
        if self.use_tencrop:
            # self.default_transform.append(T.TenCrop(192,192))
            self.default_transform.append(T.TenCrop(384,384))
            # self.default_transform.append(T.CenterCrop((576,576)))
            self.default_transform.append(T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])))
            self.default_transform.append(T.Lambda(lambda crops: torch.stack([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])))
        else:
            # self.default_transform.append(T.CenterCrop((192,192)))
            self.default_transform.append(T.CenterCrop((384,384)))
            # self.default_transform.append(T.CenterCrop((576,576)))
            self.default_transform.append(T.ToTensor())
            self.default_transform.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.default_transform = T.Compose(self.default_transform)

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, i):
        image = self.load_image(self.paths[i])
        # if self.test == False:
        #     label = torch.tensor(np.argmax(self.labels.loc[i,
        #                                    :].values))  # loss function used later doesnt take one-hot encoded labels, so convert it using argmax
        if self.test == False and self.use_onehot == False:
            label = torch.tensor(np.argmax(self.labels.loc[i,
                                           :].values))  # loss function used later doesnt take one-hot encoded labels, so convert it using argmax
        elif self.test == False and self.use_onehot == True:
            label = torch.tensor(self.labels.loc[i,:].values.astype(np.float)) 
        if self.train:
            image = self.transform(image)
        else:
            image = self.default_transform(image)
        
        # image =self.expand(image)
        # cv2.imwrite('vis_expand.jpg', image)

        if self.test == False:
            # return image, label
            return image, label, self.paths[i]
        return image

    def load_image(self, path):
        image = cv2.imread(path)
        # print(path,'\n', image)
        # import ipdb;ipdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    
    def expand(self, image):
        scale = random.uniform(0.8,1.2)
        angle = 0.
        width, height = image.shape[1], image.shape[0]
    
        cx = 0.5 * width
        cy = 0.5 * height
        
        # if randrf(0, 1) > 0.5:
        #     angle = randrf(-30, 30)
    
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        # M[0, 2] -= cx - width * 0.5
        # M[1, 2] -= cy - height * 0.5      
        image = cv2.warpAffine(image, M, (width, height))
    
        return image