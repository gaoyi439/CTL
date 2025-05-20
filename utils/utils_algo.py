import math
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import numpy as np
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def predict(Outputs, threshold):
    sig = nn.Sigmoid()
    # pre = sig(Outputs)
    pre_label = sig(Outputs)
    min_pre_label, _ = torch.min(pre_label.data, 1)
    max_pre_label, _ = torch.max(pre_label, 1)
    min_pre_label = min_pre_label.view(-1, 1)
    max_pre_label = max_pre_label.view(-1, 1)
    for i in range(len(pre_label)):
        if max_pre_label[i] - min_pre_label[i] != 0:
            pre_label[i, :] = (pre_label[i, :] - min_pre_label[i]) / (max_pre_label[i] - min_pre_label[i])
    pre = pre_label
    # pre[pre > 0.45] = 1
    # pre[pre <= 0.45] = 0
    pre[pre > threshold] = 1
    pre[pre <= threshold] = 0
    return pre

#####################################################################################################
def mcll_loss_bce(outputs, com_labels):#mcll_loss_bce
    n, K = com_labels.size()[0], com_labels.size()[1]
    comp_num = com_labels.sum(dim=1)
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(pos_outputs * torch.log(sig_outputs + 1e-12), dim=1)
    part_2 = -torch.sum(pos_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1)
    part_3 = -torch.sum(neg_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1)
    total_loss = 1.0/(2**K-2) * (2**(K-comp_num - 1)*part_1 + (2**(K-comp_num - 1)-1) * part_2 + (2**(K-comp_num)-1)*part_3)
    ave_loss = total_loss.mean()
    return ave_loss

def mcll_loss_mae(outputs, com_labels):
    n, K = com_labels.size()[0], com_labels.size()[1]
    comp_num = com_labels.sum(dim=1)
    sig = nn.Sigmoid()
    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = torch.sum(pos_outputs * (1-sig_outputs), dim=1)
    part_2 = torch.sum(pos_outputs * sig_outputs, dim=1)
    part_3 = torch.sum(neg_outputs * sig_outputs, dim=1)
    total_loss = 1.0/(2**K-2) * (2**(K-comp_num - 1)*part_1 + (2**(K-comp_num - 1)-1) * part_2 + (2**(K-comp_num)-1)*part_3)
    ave_loss = total_loss.mean()
    return ave_loss

def mcll_loss_ctl(outputs, com_labels):
    sig = nn.Sigmoid()
    loss_pre_label = sig(outputs)

    loss_min_pre_label, _ = torch.min(loss_pre_label.data, dim=1, keepdim=True)  # 保持维度以便广播
    loss_max_pre_label, _ = torch.max(loss_pre_label.data, dim=1, keepdim=True)
    ranges = loss_max_pre_label - loss_min_pre_label

    normalized = (loss_pre_label.data - loss_min_pre_label) / (ranges + 1e-12)
    loss_pre = torch.where(ranges != 0, normalized, loss_pre_label.data)

    w = 1 - com_labels
    w = w * loss_pre
    w[w > 0.3] = 0
    w[w != 0] = 1

    sig_outputs = sig(outputs)
    pos_outputs = 1 - com_labels
    neg_outputs = com_labels

    part_1 = -torch.sum(pos_outputs * torch.log(sig_outputs + 1e-12), dim=1)
    part_2 = -torch.sum(w * pos_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1)
    part_3 = -torch.sum(neg_outputs * torch.log(1.0 - sig_outputs + 1e-12), dim=1)
    total_loss = (math.e ** (-0.5 ** com_labels.sum(dim=1))) * part_1 + (
                math.e ** (-0.5 ** com_labels.sum(dim=1))) * part_2 + \
                 (math.e ** (0.5 ** (com_labels.sum(dim=1) - 1))) * part_3
    ave_loss = total_loss.mean()
    return ave_loss