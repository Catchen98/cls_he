# -*- encoding: utf-8 -*-
'''
@File    :   attention.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/11/6 20:54   xin      1.0         None
'''

from torch import nn
import torch

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
        # torch.save(y,'./STN_stage1.pkl')
        if self.multiply == True:
            return x * y
        else:
            return y

class AttentionLayer(nn.Module):
    def __init__(self, channel_in, r):
        super(AttentionLayer, self).__init__()
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

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class GloRe(nn.Module):
    def __init__(self, channel):
        super(GloRe, self).__init__()
        # node num      : int(channel/4))
        # node channel  : int(channel)
        self.reduce_dim = nn.Conv2d(channel, int(channel), kernel_size=1, stride=1, bias=False)
        self.create_projection_matrix = nn.Conv2d(channel, int(channel / 4), kernel_size=1, stride=1, bias=False)
        self.GCN_step_1 = nn.Conv1d(in_channels=int(channel), out_channels=int(channel), kernel_size=1, stride=1,
                                    padding=0)
        self.GCN_step_2 = nn.Conv1d(in_channels=int(channel / 4), out_channels=int(channel / 4), kernel_size=1,
                                    stride=1, padding=0)
        self.expand_dim = nn.Conv2d(int(channel), channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x_ = self.reduce_dim(x)
        B = self.create_projection_matrix(x)
        # reduce dim
        b, c, h, w = x_.size()
        x_ = x_.view(b, c, h * w)
        # projection matrix
        b, c, h, w = B.size()
        B = B.view(b, c, h * w)
        # b,N,L -> b,L,N
        B = torch.transpose(B, 2, 1)
        # coordinate space -> latent relation space
        V = torch.matmul(x_, B)
        # GCN_1-1
        V1 = self.GCN_step_1(V)
        V1 = torch.transpose(V1, 2, 1)
        # GCN1-2
        V2 = self.GCN_step_2(V1)
        V2 = torch.transpose(V2, 2, 1)
        # Reverse Projection Matrix
        B = torch.transpose(B, 2, 1)
        # latent relation space -> coordinate space
        Y = torch.matmul(V2, B)
        b, c, _ = Y.size()
        # b,c,numpix -> b,c,h,w
        Y = Y.view(b, c, h, w)
        # self.expand_dim
        Y = self.expand_dim(Y)

        x = x + Y
        return x