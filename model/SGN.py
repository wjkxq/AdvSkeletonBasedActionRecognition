# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math


# class SGN(nn.Module):
#     def __init__(self, num_classes=60, seg=20, train_mode=False, bias=True):
#         super(SGN, self).__init__()
#
#         self.dim1 = 256
#         # self.dataset = dataset
#         self.seg = seg
#         num_joint = 25
#         bs = 64 * 2 # 2
#         self.train_mode = train_mode
#         if self.train_mode:
#             self.spa = self.one_hot(bs, num_joint, self.seg)
#             self.spa = self.spa.permute(0, 3, 2, 1).cuda()
#             self.tem = self.one_hot(bs, self.seg, num_joint)
#             self.tem = self.tem.permute(0, 3, 1, 2).cuda()
#         else:
#             self.spa = self.one_hot(bs, num_joint, self.seg)
#             self.spa = self.spa.permute(0, 3, 2, 1).cuda()
#             self.tem = self.one_hot(bs, self.seg, num_joint)
#             self.tem = self.tem.permute(0, 3, 1, 2).cuda()
#
#         self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
#         self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
#         self.joint_embed = embed(3, 64, norm=True, bias=bias)
#         self.dif_embed = embed(3, 64, norm=True, bias=bias)
#         self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
#         self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
#         self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
#         self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
#         self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
#         self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
#         self.fc = nn.Linear(self.dim1 * 2, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#
#         nn.init.constant_(self.gcn1.w.cnn.weight, 0)
#         nn.init.constant_(self.gcn2.w.cnn.weight, 0)
#         nn.init.constant_(self.gcn3.w.cnn.weight, 0)
#
#     def forward(self, input, target=None):
#
#         # Dynamic Representation
#         # bs, step, dim = input.size()
#         N, C, T, V, M = input.shape
#         # print(input.size())
#         bs, step, dim = N, T, V * C * M
#         num_joints = dim // (C * M)
#         input = input.view((bs * M, step, num_joints, 3))
#         bs = bs * M
#         input = input.permute(0, 3, 2, 1).contiguous()
#         dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
#         dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
#         pos = self.joint_embed(input)
#         tem1 = self.tem_embed(self.tem)
#         spa1 = self.spa_embed(self.spa)
#         dif = self.dif_embed(dif)
#         dy = pos + dif
#         # Joint-level Module
#         # print(spa1.size())  # 64 64 25 60
#
#         # print(dy.size(), spa1.size())
#         input = torch.cat([dy, spa1], 1)
#         g = self.compute_g1(input)
#         input = self.gcn1(input, g)
#         input = self.gcn2(input, g)
#         input = self.gcn3(input, g)
#         # Frame-level Module
#         input = input + tem1
#         input = self.cnn(input)
#         # Classification
#         output = self.maxpool(input)
#         output = output.view(N, output.size()[1], M, 1, 1)
#         output = output.mean(2)
#         # print(output.size())
#         output.view(N, output.size()[1], 1, 1)
#         # print(output.size())
#         output = torch.flatten(output, 1)
#         output = self.fc(output)
#
#         return output
#
#     def one_hot(self, bs, spa, tem):
#
#         y = torch.arange(spa).unsqueeze(-1)
#         y_onehot = torch.FloatTensor(spa, spa)
#
#         y_onehot.zero_()
#         y_onehot.scatter_(1, y, 1)
#
#         y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
#         y_onehot = y_onehot.repeat(bs, tem, 1, 1)
#
#         return y_onehot
#
#
# class norm_data(nn.Module):
#     def __init__(self, dim=64):
#         super(norm_data, self).__init__()
#
#         self.bn = nn.BatchNorm1d(dim * 25)
#
#     def forward(self, x):
#         bs, c, num_joints, step = x.size()
#         x = x.view(bs, -1, step)
#         x = self.bn(x)
#         x = x.view(bs, -1, num_joints, step).contiguous()
#         return x
#
#
# class embed(nn.Module):
#     def __init__(self, dim=3, dim1=128, norm=True, bias=False):
#         super(embed, self).__init__()
#
#         if norm:
#             self.cnn = nn.ModuleList([
#                 norm_data(dim),
#                 cnn1x1(dim, 64, bias=bias),
#                 nn.ReLU(),
#                 cnn1x1(64, dim1, bias=bias),
#                 nn.ReLU(),
#             ])
#         else:
#             self.cnn = nn.ModuleList([
#                 cnn1x1(dim, 64, bias=bias),
#                 nn.ReLU(),
#                 cnn1x1(64, dim1, bias=bias),
#                 nn.ReLU(),
#             ])
#
#     def forward(self, x):
#         for i in range(len(self.cnn)):
#             x = self.cnn[i](x)
#         # x = self.cnn(x)
#         return x
#
#
# class cnn1x1(nn.Module):
#     def __init__(self, dim1=3, dim2=3, bias=True):
#         super(cnn1x1, self).__init__()
#         self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.cnn(x)
#         return x
#
#
# class local(nn.Module):
#     def __init__(self, dim1=3, dim2=3, bias=False):
#         super(local, self).__init__()
#         self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
#         self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
#         self.bn1 = nn.BatchNorm2d(dim1)
#         self.relu = nn.ReLU()
#         self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
#         self.bn2 = nn.BatchNorm2d(dim2)
#         self.dropout = nn.Dropout2d(0.2)
#
#     def forward(self, x1):
#         x1 = self.maxpool(x1)
#         x = self.cnn1(x1)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.cnn2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         return x
#
#
# class gcn_spa(nn.Module):
#     def __init__(self, in_feature, out_feature, bias=False):
#         super(gcn_spa, self).__init__()
#         self.bn = nn.BatchNorm2d(out_feature)
#         self.relu = nn.ReLU()
#         self.w = cnn1x1(in_feature, out_feature, bias=False)
#         self.w1 = cnn1x1(in_feature, out_feature, bias=bias)
#
#     def forward(self, x1, g):
#         x = x1.permute(0, 3, 2, 1).contiguous()
#         x = g.matmul(x)
#         x = x.permute(0, 3, 2, 1).contiguous()
#         x = self.w(x) + self.w1(x1)
#         x = self.relu(self.bn(x))
#         return x
#
#
# class compute_g_spa(nn.Module):
#     def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
#         super(compute_g_spa, self).__init__()
#         self.dim1 = dim1
#         self.dim2 = dim2
#         self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
#         self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1):
#         g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
#         g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
#         g3 = g1.matmul(g2)
#         g = self.softmax(g3)
#         return g


class SGN(nn.Module):
    def __init__(self, num_classes=60, seg=20, batch_size=64, train_mode=True, bias=True):
        super(SGN, self).__init__()

        self.dim1 = 256
        # self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = batch_size # 2
        print(bs)
        self.train_mode = train_mode
        if self.train_mode:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(1, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64 * 4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        # bone
        # self.tem_embed_bone = embed(self.seg, 64 * 4, norm=False, bias=bias)
        # self.spa_embed_bone = embed(num_joint, 64, norm=False, bias=bias)
        # self.joint_embed_bone = embed(3, 64, norm=True, bias=bias)
        # self.dif_embed_bone = embed(3, 64, norm=True, bias=bias)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        # self.maxpool_bone = nn.AdaptiveMaxPool2d((1, 1))

        self.compute_g1 = compute_g_spa(self.dim1//2, self.dim1//2, bias=bias)
        self.gcn1 = gcn_spa(self.dim1//2, self.dim1//2, bias=bias)
        self.compute_g2 = compute_g_spa(self.dim1//2, self.dim1, bias=bias)
        self.gcn2 = gcn_spa(self.dim1//2, self.dim1, bias=bias)
        self.compute_g3 = compute_g_spa(self.dim1, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        # bone
        # self.compute_g1_bone = compute_g_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        #
        # self.gcn1_bone = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
        # self.gcn2_bone = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        # self.gcn3_bone = gcn_spa(self.dim1, self.dim1, bias=bias)

        self.residual = cnn1x1(self.dim1//2, self.dim1)
        self.bn_residual = nn.BatchNorm2d(self.dim1)
        # bone
        # self.residual_bone = cnn1x1(self.dim1 // 2, self.dim1)
        # self.bn_residual_bone = nn.BatchNorm2d(self.dim1)

        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
        # bone
        # self.cnn_bone = local(self.dim1, self.dim1 * 2, bias=bias)

        self.fc = nn.Linear(self.dim1 * 2, num_classes)
        # self.fc_bone = nn.Linear(self.dim1 * 2, num_classes)
        # attention
        # self.spa_attention = spa_attention(self.dim1)
        self.tem_attention = tem_attention(self.dim1)
        # self.tem_attention_bone = tem_attention(self.dim1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.w.cnn1.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn1.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn1.weight, 0)


    def forward(self, input, target=None):

        # Dynamic Representation
        # 64, 20, 75
        N, C, T, V, M = input.shape
        # print('batch size: ',N)

        bs, step, dim = N, T, V * C * M
        num_joints = dim // (C * M)
        input = input[:, :, :, :, 0]
        # print(input.size())
        input = input.view((bs, step, num_joints, 3))
        bs = bs

        input = input.permute(0, 3, 2, 1).contiguous()
        # bone
        # input_bone = self.gen_bone(input)

        # velocity
        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)
        # position
        pos = self.joint_embed(input)
        tem1 = self.tem_embed(self.tem)

        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)
        # print(pos.size(), dif.size())
        dy = pos + dif

        # bone - velocity
        # dif_bone = input_bone[:, :, :, 1:] - input_bone[:, :, :, 0:-1]
        # dif_bone = torch.cat([dif_bone.new(bs, dif_bone.size(1), num_joints, 1).zero_(), dif_bone], dim=-1)
        # # position
        # pos_bone = self.joint_embed_bone(input_bone)
        # tem1_bone = self.tem_embed_bone(self.tem)
        # spa1_bone = self.spa_embed_bone(self.spa)
        # dif_bone = self.dif_embed_bone(dif_bone)
        # dy_bone = pos_bone + dif_bone

        # Joint-level Module
        # print(dy.size(), spa1.size())
        # print(dy.size(), spa1.size())
        input = torch.cat([dy, spa1], 1)
        # input_bone = torch.cat([dy_bone, spa1_bone], 1)
        # input = torch.cat([input, input_bone], 1)
        # [64, 256, 25, 20]

        # g -- 经过softmax归一化的代表关节点直接联系的邻接矩阵
        g1 = self.compute_g1(input)
        input1 = input
        input = self.gcn1(input, g1)
        g2 = self.compute_g2(input)
        g2 = g2 + g1
        input = self.gcn2(input, g2)
        g3 = self.compute_g3(input)
        g3 = g3 + g2
        input = self.gcn3(input, g3)
        input = input + self.bn_residual(self.residual(input1))
        # 64, 256, 25, 20

        #


        # Frame-level Module
        # tem1 = torch.cat([tem1, tem1_bone], 1)
        input = input + tem1
        # input = self.spa_attention(input)
        input = self.tem_attention(input)
        input = self.cnn(input)

        # Classification
        output = self.maxpool(input)
        # bone
        # output_bone = self.maxpool_bone(input_bone)


        # 将多维数据展开，1--保持第一维不变
        # output = output + output_bone
        output = torch.flatten(output, 1)
        # output_bone = torch.flatten(output_bone, 1)
        output = self.fc(output)
        # output_bone = self.fc_bone(output_bone)
        # 64, 60
        # print(output[0])不全为0

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    def gen_bone(self, input):
        # input = input.permute(0, 3, 2, 1).contiguous()
        n, c, v, t = input.size()
        input1 = input[:, :c, :, :]
        if self.case == 0:
            mode = 'ntu/cs'
        else:
            mode = 'ntu/cv'
        for v1, v2 in paris[mode]:
            v1 -= 1
            v2 -= 1
            input1[:, :, v1, :] = input[:, :, v1, :] - input[:, :, v2, :]
        return input1


class norm_data(nn.Module):
    def __init__(self, dim=64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim * 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x


class embed(nn.Module):
    def __init__(self, dim=3, dim1=128, norm=True, bias=False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x


class cnn1x1(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=True):
        super(cnn1x1, self).__init__()
        # self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        # self.shift_in = Shift(channel=dim1, stride=1, init_scale=1)
        self.cnn1 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

        # self.shift_out = Shift(channel=dim2, stride=1, init_scale=1)

    def forward(self, x):
        x = self.cnn1(x)
        return x


class local(nn.Module):
    def __init__(self, dim1=3, dim2=3, bias=False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        # self.shift_in = Shift(channel=dim1, stride=1, init_scale=1)
        self.bn1_1 = nn.BatchNorm2d(dim1)
        # self.bn1_2 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        # self.shift_out = Shift(channel=dim2, stride=1, init_scale=1)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

        self.residual = cnn1x1(dim1, dim2)
        self.bn_re = nn.BatchNorm2d(dim2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x0 = x1
        # CNN-1
        # x = self.bn1_1(x1)
        x = self.cnn1(x1)
        # x = self.shift_in(x1)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # CNN-2
        # x = self.bn1_2(x)
        x = self.cnn2(x)
        # x = self.shift_out(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x + self.bn_re(self.residual(x0))


class spa_attention(nn.Module):
    def __init__(self, out_channels, num_joints=25):
        super(spa_attention, self).__init__()
        # spatial attention
        ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        self.fai = cnn1x1(out_channels, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, y):
        y1 = y
        y2 = self.fai(y1)

        # spatial attention
        se = y.mean(-1)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-1) + y

        return self.bn(y2 + y)


class tem_attention(nn.Module):
    def __init__(self, out_channels):
        super(tem_attention, self).__init__()

        # temporal attention
        # self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        # nn.init.constant_(self.conv_ta.weight, 0)
        # nn.init.constant_(self.conv_ta.bias, 0)
        #
        # self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(out_channels)
        self.att = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, y):
        y1 = self.att(y)
        return self.bn(y1)
        # y1 = y
        # temporal attention
        # se = y.mean(-2)
        # se1 = self.sigmoid(self.conv_ta(se))
        # y = y * se1.unsqueeze(-2) + y
        # return self.bn(y + y1)




class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias=False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)
        self.spa_att = spa_attention(out_feature)

    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))

        # spa attention
        x = self.spa_att(x)
        return x


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)




class compute_g_spa(nn.Module):
    def __init__(self, dim1=64 * 3, dim2=64 * 3, bias=False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        # self.spa_attention1 = spa_attention(self.dim1)
        # self.spa_attention2 = spa_attention(self.dim1)

        # new
        # num_joints = 25
        # ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        # pad = (ker_jpt - 1) // 2
        # self.conv_sa1 = nn.Conv1d(dim1, 1, ker_jpt, padding=pad)
        # # self.conv_sa2 = nn.Conv1d(dim1, 1, ker_jpt, padding=pad)
        #
        # nn.init.xavier_normal_(self.conv_sa1.weight)
        # # nn.init.xavier_normal_(self.conv_sa2.weight)
        # nn.init.constant_(self.conv_sa1.bias, 0)
        # # nn.init.constant_(self.conv_sa2.bias, 0)
        # self.sigmoid1 = nn.ReLU()
        # self.sigmoid2 = nn.Sigmoid()

        # tem
        # self.conv_ta = nn.Conv1d(dim1, 1, 9, padding=4)
        # nn.init.constant_(self.conv_ta.weight, 0)
        # nn.init.constant_(self.conv_ta.bias, 0)

        # self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(dim2)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        g1 = self.bn1(self.g1(x1)).permute(0, 3, 2, 1).contiguous()
        g2 = self.bn2(self.g2(x1)).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)

        # x2 = x1

        # se1 = x1.mean(-1)  # N C V
        # se11 = self.sigmoid1(self.conv_sa1(se1))
        # # bn
        # x1 = self.bn1(x1 * se11.unsqueeze(-1))
        # x1 = x1.permute(0, 3, 1, 2).contiguous()

        # se2 = x2.mean(-1)  # N C V
        # se22 = self.sigmoid2(self.conv_sa2(se2))
        # x2 = self.bn2(x2 * se22.unsqueeze(-1))
        # x2 = x2.permute(0, 3, 1, 2).contiguous()

        # tem
        # se = x2.mean(-2)
        # se1 = self.sigmoid(self.conv_ta(se))
        # x2 = x2 * se1.unsqueeze(-2)
        # x2 = x2.permute(0, 3, 1, 2).contiguous()

        # g = g1.matmul(x1)
        # g = self.softmax(g)
        return g


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
