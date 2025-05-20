import argparse
import os

import torch.nn as nn
import torch
import numpy as np
from torch.backends import cudnn
import random

from utils import dataset
from utils.metrics import OneError, Coverage, HammingLoss, RankingLoss, AveragePrecision
from utils.models import linear, co_linear, MLP
from utils.utils_algo import adjust_learning_rate, predict,mcll_loss_bce,mcll_loss_mae,mcll_loss_ctl
from utils.utils_data_mcl import choose

parser = argparse.ArgumentParser(description='PyTorch implementation of CTL')
parser.add_argument('--dataset', default='scene', type=str, help='dataset name')
parser.add_argument('--num-class', default=6, type=int, help='number of classes')
parser.add_argument('--input-dim', default=294, type=int, help='number of features')
parser.add_argument('--fold', default=9, type=int, help='fold-th fold of 10-cross fold')
parser.add_argument('--model', default="LeNet", type=str, choices=['MLP', 'linear'])
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--schedule', default=[100, 150], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--lo', default="sigmoid", type=str)
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--the', default=0.8, type=float, help='seed for initializing training. ')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # print(args)

    cudnn.benchmark = True

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # make data
    train_loader, test_loader, args.num_class, args.input_dim = choose(args)

    # choose model
    if args.model == "linear":
        model = linear(input_dim=args.input_dim, output_dim=args.num_class)
    elif args.model == "MLP":
        model = MLP(input_dim=args.input_dim, hidden_dim=500, output_dim=args.num_class)

    model = model.to(device)

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.wd)

    criterion = nn.BCEWithLogitsLoss()

    print("start training")

    best_av = 0
    save_table = np.zeros(shape=(args.epochs, 7))
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train(train_loader, model, optimizer, args, criterion)
        t_hamm, t_one_error, t_converage, t_rank, t_av_pre = validate(test_loader, model, args)
        print("Epoch:{ep}, Tr_loss:{tr}, T_hamm:{T_hamm}, T_one_error:{T_one_error}, T_con:{T_con}, "
              "T_rank:{T_rank}, T_av:{T_av}".format(ep=epoch, tr=train_loss, T_hamm=t_hamm, T_one_error=t_one_error,
                                                    T_con=t_converage, T_rank=t_rank, T_av=t_av_pre))
        save_table[epoch, :] = epoch + 1, train_loss, t_hamm, t_one_error, t_converage, t_rank, t_av_pre

        if not os.path.exists('./result/ctl/'):
            os.makedirs('./result/ctl/')

        np.savetxt("./result/ctl/{ds}_{md}_{M}_lr{lr}_wd{wd}_fold{fd}.csv".format(name=args.dataset, ds=args.dataset, md=args.lo,
                                                                                  M=args.model, lr=args.lr, wd=args.wd,
                                                                                  fd=args.fold), save_table,
                   delimiter=',', fmt='%1.4f')
        # save model

        if t_av_pre > best_av:
            best_av = t_av_pre

            if not os.path.exists('./result/ctl/'):
                os.makedirs('./result/Linear/ctl/')
            torch.save(model.state_dict(), "./result/ctl/{ds}_{md}_{M}_lr{lr}_wd{wd}_fold{fd}_best_model.tar".format(
                ds=args.dataset, md=args.lo, M=args.model, lr=args.lr, wd=args.wd, fd=args.fold))



def train(train_loader, model, optimizer, args, criterion):
    model.train()
    train_loss = 0
    for i, (images, _, com_labels, _, index) in enumerate(train_loader):
        images, com_labels = images.to(device), com_labels.to(device)

        com_labels = com_labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        if args.lo == "mcll_bce":
            loss = mcll_loss_bce(outputs, com_labels)
        elif args.lo == "mcll_mae":
            loss = mcll_loss_mae(outputs, com_labels)
        elif args.lo == "mcll_ctl":
            loss = mcll_loss_ctl(outputs, com_labels)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()

    return train_loss / len(train_loader)


# test the results
def validate(test_loader, model, args):
    with torch.no_grad():
        model.eval()
        sig = nn.Sigmoid()
        t_one_error = 0
        t_converage = 0
        t_hamm = 0
        t_rank = 0
        t_av_pre = 0

        for data, targets, _, _, _ in test_loader:
            images, targets = data.to(device), targets.to(device)
            output = model(images)
            pre_output = sig(output)
            pre_label = predict(output, args.the)

            t_one_error = t_one_error + OneError(pre_output, targets)
            t_converage = t_converage + Coverage(pre_output, targets)
            t_hamm = t_hamm + HammingLoss(pre_label, targets)
            t_rank = t_rank + RankingLoss(pre_output, targets)
            t_av_pre = t_av_pre + AveragePrecision(pre_output, targets)

    return t_hamm/len(test_loader), t_one_error/len(test_loader), t_converage/len(test_loader), t_rank/len(test_loader), \
        t_av_pre/len(test_loader)


if __name__ == '__main__':
    data = ["scene_uniform"]
    lr_1e_1 = ["Corel16k15_uniform","rcv2_15_uniform", "rcv3_15_uniform", "rcv4_15_uniform"]
    lr_1e_2 = ["scene_uniform","ml_tmc2007_uniform"]
    lr_1e_3 = ["VOC2007_uniform"]

    theo = [0.5]
    for i in theo:
        for fd in range(10):
            for ds in data:
                args = parser.parse_args()

                if ds in lr_1e_1:
                    args.lr = 0.1
                elif ds in lr_1e_2:
                    args.lr = 0.01
                else:
                    args.lr = 0.001

                args.dataset = ds
                args.fold = fd
                args.the = i
                args.model = 'linear'

                args.lo = "mcll_ctl"
                print(args)
                main()