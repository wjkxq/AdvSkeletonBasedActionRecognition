# encoding: utf-8

"""
@author: huguyuehuhu
@time: 18-4-16 下午6:51
Permission is given to modify the code, any problem please contact huguyuehuhu@gmail.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import utils
import torchvision
import os

class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self, in_channel=3, num_joint=25, num_person=2, out_channel=64, window_size=64,
                 num_class=60,
                 ):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=(1,0))

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        # 分类边界确定
        self.fc7= nn.Sequential(
            nn.Linear((out_channel * 4)*(window_size//16)*(window_size//16), 256*2), # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        self.fc9 = nn.Sequential(
            nn.Linear((out_channel * 4) * (window_size // 16) * (window_size // 16), 256 * 2),
            # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc10 = nn.Linear(256 * 2, num_class)

        # initial weight
        utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        # print(x.size())
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1:,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        logits = []
        for i in range(self.num_person):
            # position
            # N0,C1,T2,V3 point-level
            out = self.conv1(x[:,:,:,:,i])

            out = self.conv2(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0,3,2,1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
            out = self.conv1m(motion[:,:,:,:,i])
            out = self.conv2m(out)
            # N0,V1,T2,C3, global level
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # concat
            out = torch.cat((out_p,out_m),dim=1)
            out = self.conv5(out)
            out = self.conv6(out)

            logits.append(out)

        # max out logits
        out = torch.max(logits[0], logits[1])
        out = out.view(out.size(0), -1)
        # 分类边界
        tmp = out
        out = self.fc7(out)

        out = self.fc8(out)

        tmp = self.fc9(tmp)

        tmp = self.fc10(tmp)

        t = (out + tmp) * 0.5
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor


        return out


def loss_fn(outputs,labels,current_epoch=None,params=None):
    """
    Compute the cross entropy loss given outputs and labels.

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    if params.loss_args["type"] == 'CE':
        CE = nn.CrossEntropyLoss()(outputs, labels)
        loss_all = CE
        loss_bag = {'ls_all': loss_all, 'ls_CE': CE}
    # elif: other losses

    return loss_bag


def acc(output, target):
    _, output = output.topk(1, 1, True, True)
    output = output.squeeze()

    right = 0
    for i in range(len(output)):
        if output[i].data.item() == target[i].data.item():
            right += 1
    print('acc', right/len(output))
    return right/len(output)


def accuracytop1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(pred.size(), target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).data.item())
    return res

def accuracytop2(output, target, topk=(2,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).data.item())
    return res

def accuracytop3(output, target, topk=(3,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).data.item())
    return res

def accuracytop5(output, target, topk=(5,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size).data.item())
    return res


def avarage_confidence(output, target):
    avg = 0
    batch_size = len(output)
    output = F.softmax(output, dim=1)
    for out in output:
        # avg += max(out).data.item()
        avg += out[0].data.item()
    # print('AC: ', avg/batch_size)
    return avg/batch_size


def avg_conf_of_true(output, target):

    batch_size = len(output)
    output = F.softmax(output, dim=1)
    avg = 0

    for i in range(len(output)):
        # print(target)
        avg += output[i][target[i]].data.item()
    # print('ACTC: ', avg/batch_size)
    return avg/batch_size


def robust_to_noise(output, target):
    batch_size = len(output)
    output = F.softmax(output, dim=1)
    avg = 0
    for o in output:
        max1, max2 = 0, 0
        for i in o:
            if i.data.item()>max2:
                max2 = i.data.item()
            if max2 > max1:
                max1, max2 = max2, max1
        # print(max1, max2)
        avg += (max1 - max2)
    # print('RN: ', avg/batch_size)
    return avg/batch_size


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop5': accuracytop5,
    'acc': acc,
    'AC': avarage_confidence,
    'ACTC': avg_conf_of_true,
    'RN': robust_to_noise
    # could add more metrics such as accuracy for each token type
}

from thop import profile
import thop

if __name__ == '__main__':
    model = HCN().cuda()
    # children = list(model.children())
    # print(children)
    input = torch.randn(1, 3, 32, 25, 2).cuda()
    flops, params = profile(model, inputs=(input,))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
