# -*- encoding: utf-8 -*-
'''
@File    :   attention.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 20:54   xin      1.0         None
'''
import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=64, multiply=True):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
                )
        self.multiply = multiply
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        if self.multiply == True:
            return x * y
        else:
            return y


class STNLayer(nn.Module):
    def __init__(self, channel_in, multiply=True):
        super(STNLayer, self).__init__()
        c = channel_in
        C = c//32
        self.multiply = multiply
        self.conv_in = nn.Conv2d(c, C, kernel_size=1)
        self.conv_out = nn.Conv2d(C, 1, kernel_size=1)
        # Encoder
        self.conv1 = nn.Conv2d(C, 2*C, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(2*C)
        self.ReLU1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(2*C, 4*C, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(4*C)
        self.ReLU2 = nn.ReLU(True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(4*C, 2*C, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2*C)
        self.ReLU3 = nn.ReLU(True)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(2*C, C, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(C)
        self.ReLU4 = nn.ReLU(True)


    def forward(self, x):
        b, c, _, _ = x.size()
        #print("modules: x.shape: " + str(x.shape))
        y = self.conv_in(x)

        # Encode
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.ReLU1(y)
        size1 = y.size()
        y, indices1 = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.ReLU2(y)

        # Decode
        y = self.deconv1(y)
        y = self.bn3(y)
        y = self.ReLU3(y)
        y = self.unpool1(y,indices1,size1)
        y = self.deconv2(y)
        y = self.bn4(y)
        y = self.ReLU4(y)

        y = self.conv_out(y)
        #torch.save(y,'./STN_stage1.pkl')
        if self.multiply == True:
            return x * y
        else:
            return y


class SESTNLayer(nn.Module):
    def __init__(self, channel_in, r):
        super(SESTNLayer, self).__init__()
        c = channel_in
        self.se = SELayer(channel=c, reduction=r, multiply=False)
        self.stn = STNLayer(channel_in=c, multiply=False)
        self.activation = nn.Hardtanh(inplace=True)
        self.activation = nn.ReLU(True)


    def forward(self, x):
        y = self.se(x)
        z = self.stn(x)
        a = self.activation(y+z) # Final joint attention map
        return x + x*a

class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()
    def forward(self,  x, weights):
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
    def __repr__(self):
        return self.__class__.__name__ + '()'


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.in_c = in_c
        self.act_fn = act_fn
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)                 # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)                    # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=20)       # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
        + 'in_c={}'.format(self.in_c) \
        + 'act_fn={}'.format(self.act_fn) \
        +')'