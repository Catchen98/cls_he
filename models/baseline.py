# -*- encoding: utf-8 -*-
'''
@File    :   baseline.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 18:14   xin      1.0         None
'''

import torch
from torch import nn

from .backbones.resnet import ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnext_ibn_a import resnext101_ibn_a
from .backbones.resnest import *
from .backbones.senet import * 
from efficientnet_pytorch import EfficientNet
from .layers.pooling import GeM,GlobalConcatPool2d,GlobalAttnPool2d,GlobalAvgAttnPool2d,GlobalMaxAttnPool2d,GlobalConcatAttnPool2d,AdaptiveGeM2d

def weights_init_kaiming(m):
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


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path=None, backbone="resnet50", pool_type='avg', use_dropout=True, use_bnneck=True, cls_bias=False, use_bnbias=False):
        super(Baseline, self).__init__()
        if backbone == "resnet50":
            self.base = ResNet(last_stride)
            # self.base.load_param(model_path)
        elif backbone == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride)
        elif 'efficientnet' in backbone:
            self.base = EfficientNet.from_pretrained(backbone)
        else:
            self.base = eval(backbone)(last_stride=last_stride)
        if 'efficientnet' not in backbone:
            self.base.load_param(model_path)
        in_features = self.in_planes

        self.backbone = backbone
        if backbone == 'efficientnet-b0':
            in_features = 1280
        elif backbone == 'efficientnet-b6':
            in_features = 2304
        elif backbone == 'efficientnet-b7':
            in_features = 2560
            
        if pool_type == "avg":
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif "gem" in pool_type:
            if pool_type !='gem':
                p = pool_type.split('_')[-1]
                p = float(p)
                self.gap = AdaptiveGeM2d(p=p, eps=1e-6, freeze_p=True, clip_max=1e8)
            else:
                self.gap = AdaptiveGeM2d(eps=1e-6, freeze_p=False, clip_max=1e8) # clip_max to support convert2keras
        elif pool_type == 'max':
            self.gap = nn.AdaptiveMaxPool2d(1)
        elif 'Att' in pool_type:
            self.gap = eval(pool_type)(in_features = in_features)
            in_features = self.gap.out_features(in_features)
        else:
            self.gap = eval(pool_type)()
            in_features = self.gap.out_features(in_features)

        self.num_classes = num_classes
        self.use_bnneck = use_bnneck

        if self.use_bnneck:
            self.bottleneck = nn.BatchNorm1d(in_features)
            self.bottleneck.bias.requires_grad_(use_bnbias)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(in_features, self.num_classes, bias=cls_bias)
        self.classifier.apply(weights_init_classifier)
        # self.dropout = torch.nn.Dropout()
        self.use_dropout = use_dropout
    
        # if self.dropout >= 0:
        if use_dropout:
            # self.dropout = nn.Dropout(self.dropout)
            self.dropout = nn.Dropout(0.5)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
    def forward(self, x, ret_score=False):
        if 'efficientnet' in self.backbone:
            global_feat = self.gap(self.base.extract_features(x))  # (b, 1280, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 1280)
        else:
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], 2048)  # flatten to (bs, 2048)
        if self.use_bnneck:
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        else:
            feat = global_feat
        if self.use_dropout:
            feat = self.dropout(feat)
        cls_score = self.classifier(feat)
        return cls_score
        # if self.training:
        #     if self.use_dropout:
        #         feat = self.dropout(feat)
        #     cls_score = self.classifier(feat)
        #     print('------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if ret_score:
        #         if self.use_dropout:
        #             feat = self.dropout(feat)
        #         cls_score = self.classifier(feat)
        #         return cls_score, feat
        #     print('-------------------------')
        #     return feat

class Baseline_freeze(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path=None, backbone="resnet50", pool_type='avg', use_dropout=True, use_bnneck=True, cls_bias=False, use_bnbias=False):
        super(Baseline_freeze, self).__init__()
        if backbone == "resnet50":
            self.base = ResNet(last_stride)
        elif backbone == "resnet50_ibn_a":
            self.base = resnet50_ibn_a(last_stride)
        elif 'efficientnet' in backbone:
            self.base = EfficientNet.from_pretrained(backbone)
        else:
            self.base = eval(backbone)(last_stride=last_stride)
        if 'efficientnet' not in backbone:
            self.base.load_param(model_path)
        in_features = self.in_planes

        self.backbone = backbone
        if backbone == 'efficientnet-b0':
            in_features = 1280
        elif backbone == 'efficientnet-b6':
            in_features = 2304
        elif backbone == 'efficientnet-b7':
            in_features = 2560
            
        if pool_type == "avg":
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif "gem" in pool_type:
            if pool_type !='gem':
                p = pool_type.split('_')[-1]
                p = float(p)
                self.gap = AdaptiveGeM2d(p=p, eps=1e-6, freeze_p=True, clip_max=1e8)
            else:
                self.gap = AdaptiveGeM2d(eps=1e-6, freeze_p=False, clip_max=1e8) # clip_max to support convert2keras
        elif pool_type == 'max':
            self.gap = nn.AdaptiveMaxPool2d(1)
        elif 'Att' in pool_type:
            self.gap = eval(pool_type)(in_features = in_features)
            in_features = self.gap.out_features(in_features)
        else:
            self.gap = eval(pool_type)()
            in_features = self.gap.out_features(in_features)

        self.num_classes = num_classes
        self.use_bnneck = use_bnneck

        if self.use_bnneck:
            self.bottleneck = nn.BatchNorm1d(in_features)
            self.bottleneck.bias.requires_grad_(use_bnbias)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(in_features, self.num_classes, bias=cls_bias)
        self.classifier.apply(weights_init_classifier)
        # self.dropout = torch.nn.Dropout()
        self.use_dropout = use_dropout
    
        # if self.dropout >= 0:
        if use_dropout:
            # self.dropout = nn.Dropout(self.dropout)
            self.dropout = nn.Dropout(0.5)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
    def forward(self, x, ret_score=False):
        if 'efficientnet' in self.backbone:
            global_feat = self.gap(self.base.extract_features(x))  # (b, 1280, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 1280)
        else:
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], 2048)  # flatten to (bs, 2048)
        if self.use_bnneck:
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        else:
            feat = global_feat
        if self.use_dropout:
            feat = self.dropout(feat)
        cls_score = self.classifier(feat)
        return cls_score
        # if self.training:
        #     if self.use_dropout:
        #         feat = self.dropout(feat)
        #     cls_score = self.classifier(feat)
        #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if ret_score:
        #         if self.use_dropout:
        #             feat = self.dropout(feat)
        #         cls_score = self.classifier(feat)
        #         return cls_score, feat
        #     print('-------------------------')
        #     return feat