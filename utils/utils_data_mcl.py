from torch.utils.data import DataLoader
from utils import dataset
import torchvision.transforms as transforms
from scipy.special import comb
import numpy as np
import torch

def choose(args):
    if args.dataset == 'scene_uniform':
        print('Data Preparation of scene_mcl')
        file_name = ["./data/scene/scene_data.csv", "./data/scene/scene_label.csv", "./data/scene/scene_uniform_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 6
        input_dim = 294
    elif args.dataset == "ml_tmc2007_uniform":
        print('Data Preparation of ml_tmc2007_uniform_mcl')
        file_name = ["./data/ml_tmc2007/ml_tmc2007_data.csv", "./data/ml_tmc2007/ml_tmc2007_label.csv",
                     "./data/ml_tmc2007/ml_tmc2007_uniform_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 22
        input_dim = 981
    elif args.dataset == "ml_tmc2007_com":
        print('Data Preparation of ml_tmc2007_com')
        file_name = ["./data/ml_tmc2007/ml_tmc2007_data.csv", "./data/ml_tmc2007/ml_tmc2007_label.csv",
                     "./data/ml_tmc2007/ml_tmc2007_com_label.csv"]
        train_loader, test_loader = dataset.ComFold(args.batch_size, file_name, 10, args.fold)
        num_class = 22
        input_dim = 981

    return train_loader, test_loader, num_class, input_dim


