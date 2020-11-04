from pdb import set_trace
import torch
import torch.nn.functional as F
from torch import nn
from .backbones.resnest import *

from .backbones.resnet import ResNet
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.resnext_ibn_a import resnext101_ibn_a
from .layers.pooling import get_adaptive_pooling
from .layers.cosine_loss import AdaCos,ArcFace,SphereFace,CosFace,ArcCos
from .layers.attention import SpatialAttention2d, WeightedSum2d
from efficientnet_pytorch import EfficientNet


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
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def __freeze_weights__(module):
    for param in module.parameters():
        param.requires_grad = False

class DELF(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path,  backbone="resnet50", target_layer='layer3', l2_norm_att=True, use_dropout=False, cosine_loss_type='',s=30.0,m=0.35,use_bnbias=False,use_sestn=False,use_bnneck=False):
        super(DELF, self).__init__()
        if backbone == "resnet50":
            feat_extractor = ResNet(last_stride)
        elif backbone == "resnet50_ibn_a":
            feat_extractor = resnet50_ibn_a(last_stride,use_sestn=use_sestn)
        else:
            feat_extractor = eval(backbone)(last_stride=last_stride)
        # feat_extractor.load_param(model_path)
        if len(model_path):
            print('==> loading feature extractor params..')
            param_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

            start_with_module = False
            for k in param_dict.keys():
                if k.startswith('module.'):
                    start_with_module = True
                    break
            if start_with_module:
                param_dict = {k.replace('module.', '').replace('base.', ''): v for k, v in param_dict.items() }
            else:
                param_dict = {k.replace('base.', ''): v for k, v in param_dict.items() }

            print('ignore_param:')
            print([k for k, v in param_dict.items() if k not in feat_extractor.state_dict() or feat_extractor.state_dict()[k].size() != v.size()])
            print('unload_param:')
            print([k for k, v in feat_extractor.state_dict().items() if k not in param_dict.keys() or param_dict[k].size() != v.size()] )

            _param_dict = {k: v for k, v in param_dict.items() if k in feat_extractor.state_dict() and feat_extractor.state_dict()[k].size() == v.size()}
            for i in _param_dict:
                if not start_with_module:
                    feat_extractor.state_dict()[i].copy_(_param_dict[i])
                else:
                    feat_extractor.state_dict()[i].copy_(_param_dict[i.replace('module.','')])
            
        self.base = nn.Sequential(
            feat_extractor.conv1,
            feat_extractor.bn1,
            feat_extractor.relu,
            feat_extractor.maxpool,
            feat_extractor.layer1,
            feat_extractor.layer2,
            feat_extractor.layer3
        )
        self.layer4 = feat_extractor.layer4
        self.l2_norm_att = l2_norm_att
        ##[todo] freeze backbone
        __freeze_weights__(self.base)
        __freeze_weights__(self.layer4)
    
        in_features = self.in_planes

        self.target_layer = target_layer
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_bnneck = use_bnneck 
        if self.target_layer == 'layer3':
            self.rf = 291.0
            self.stride = 16.0
            self.padding = 145.0
        elif self.target_layer == 'layer4':
            self.rf = 483.0
            self.stride = 32.0
            self.padding = 241.0
        else:
            raise ValueError('Unsupported target_layer: {}'.format(self.target_layer))

        in_features = self.__get_attn_nfeats__(backbone, target_layer)
        self.attn = SpatialAttention2d(in_c=in_features, act_fn='relu')
        self.weight_pool = WeightedSum2d()

        if self.use_bnneck:
            self.bottleneck = nn.BatchNorm1d(in_features)

            if use_bnbias == False:
                print('==> remove bnneck bias')
                self.bottleneck.bias.requires_grad_(False)  # no shift
            else:
                print('==> using bnneck bias')
            self.bottleneck.apply(weights_init_kaiming)
            if len(model_path):
                print('==> load bnneck params..')
                _param_dict = {k: v for k, v in param_dict.items() if k in self.bottleneck.state_dict() and self.bottleneck.state_dict()[k].size() == v.size()}
                for i in _param_dict:
                    if not start_with_module:
                        self.bottleneck.state_dict()[i].copy_(_param_dict[i])
                    else:
                        self.bottleneck.state_dict()[i].copy_(_param_dict[i.replace('module.','')])
        if cosine_loss_type=='':
            self.classifier = nn.Linear(in_features, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        else:
            if cosine_loss_type == 'AdaCos':
                self.classifier = eval(cosine_loss_type)(in_features, self.num_classes,m)
            else:
                self.classifier = eval(cosine_loss_type)(in_features, self.num_classes,s,m)
        if len(model_path):
            print('==> load classifier params..')
            _param_dict = {k: v for k, v in param_dict.items() if k in self.classifier.state_dict() and self.classifier.state_dict()[k].size() == v.size()}
            for i in _param_dict:
                if not start_with_module:
                    self.classifier.state_dict()[i].copy_(_param_dict[i])
                else:
                    self.classifier.state_dict()[i].copy_(_param_dict[i.replace('module.','')])

        self.cosine_loss_type = cosine_loss_type
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(self.dropout)
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
    def __get_attn_nfeats__(self, arch, target_layer):
        # adjust input channels according to arch.
        in_c = 1024
        if arch in ['resnet18', 'resnet34']:
            in_c = 512
        else:
            if target_layer in ['layer3']:
                in_c = 1024
            elif target_layer in ['layer4']:
                in_c = 2048
        return in_c
    ## trainning
    def forward(self, x, label=None, ret_score=False):
        x = self.base(x)  # (b, 2048, 1, 1)
        if self.target_layer == 'layer4':
            x = self.layer4(x)
        if self.l2_norm_att:
            attn_x = F.normalize(x, p=2, dim=1)
        else:
            attn_x = x
        attn_score = self.attn(x)
        global_feat = self.weight_pool(attn_x, attn_score)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.use_bnneck:
            feat = self.bottleneck(global_feat)  
        else:
            feat = global_feat
        if self.training:
            if self.use_dropout:
                feat = self.dropout(feat)
            if self.cosine_loss_type == '':
                cls_score = self.classifier(feat)
            else:
                cls_score = self.classifier(feat,label)                
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if ret_score:
                if self.use_dropout:
                    feat = self.dropout(feat)
                cls_score = self.classifier(feat)
                return cls_score, feat
            return feat
    ## forward to extract keypoint
    def forward_for_serving(self, x):
        x = self.base(x)  # (b, 2048, 1, 1)
        if self.target_layer == 'layer4':
            x = self.layer4(x)
        attn_score = self.attn(x)
        return x.data.cpu(), attn_score.data.cpu()
