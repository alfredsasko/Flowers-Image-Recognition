'''
Module for transfer learning executes following steps
- reads passed arguments
- initializes model
- load dataset
- train & validate model
- save model
'''

# IMPORT DEPENDENCIES
# -------------------
# basics
import numpy as np
import pandas as pd
import time
import os
import copy
import warnings
warnings.simplefilter('ignore')
from collections import OrderedDict

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# data processing support
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import json

# command line support
import argparse

# keep workspace running
from workspace_utils import active_session

# custom made modules
import model
import utils

# READ ARGUMENTS
# --------------
# DEFINE ARGUMENTS
parser = argparse.ArgumentParser(description="Parser of training script")
parser.add_argument('data_dir', help='Data root directory. Mandatory argument', type=str)
parser.add_argument('--save_dir', help='Directory to save trained model. Optional argument', type=str)
parser.add_argument('--arch', help='Base model type. Any vgg or resenet model can be used. Dafault is resnet50', type=str)
parser.add_argument('--pretrained', help='Load model with pretrained weights, by default True. Optional argument', type=str)
parser.add_argument('--start_lr', help='Learning rate, default value 0.001', type=float)
parser.add_argument('--end_lr', help='Learning rate of last epoch. If specified learning rate gradually decay each epoch. Optional argument', type=float)
parser.add_argument('--hidden_layers', help='Hidden layers in classifier provided as list ex. 1023, 512. No hidden layers by default None', type=str)
parser.add_argument('--epochs', help='Number of epochs, default 10', type=int)
parser.add_argument('--gpu', help='Option to use gpu for training, cpu by default.', type=str)
parser.add_argument('--num_classes', help='Number of classes, 102 by default. Optional argument', type=str)
parser.add_argument('--feature_extract', help='Feature extaction (True by default) or tunning (False) type of transfer learning. Optional argument', type=bool)
parser.add_argument('--drop_p', help='Drop out probability, by default 0.5. Optional argument', type=float)
parser.add_argument('--batch_size', help='Size of the batch for training & validation, by default 32. Optional argument', type=int)
parser.add_argument('--avgs', help='Averages for purpose of image processing, standardization, by default 0.485, 0.456, 0.406 for each channel', type=str)
parser.add_argument('--stdvs', help='Standard deviations for purpose of image processing, standardization, by default 0.229, 0.224, 0.225 for each channel', type=str)
parser.add_argument('--resize', help='Pixel size for purpsoe of image processing resizing, by default 226 pixels. Optional argument', type=int)
parser.add_argument('--rot', help='Rotation in degrees for purpose of image processing rotation, by default 30 degrees. Optional argument', type=float)
parser.add_argument('--batch_ratio', help='Train and validate model on x% of data for tuning purposes. Saves computation power. Default 1.', type=float)

# INITIALIZE AND READ ARGUMENTS
args = parser.parse_args()

# selected models
net_name = args.arch if args.arch else 'resnet50'

# number of classes
num_classes = args.num_classes if args.num_classes else 102

# type of transfer learning: feature extraction or tunning
feature_extract = True if args.feature_extract == None or \
                  args.feature_extract == 'True' else False

# architecture of new classifier: just hidden layers sizes in the list
hidden_layers = [int(layer.strip()) for layer in args.hidden_layers.split(',')] \
                if args.hidden_layers else []

# initialization of weights of the model: pre-trained or random
pretrained = True if args.pretrained == None or \
             args.pretrained == 'True' else False

# drop out probability
drop_p = args.drop_p if args.drop_p else 0.5

# dataset root directory
data_dir = args.data_dir

# batch size for training
batch_size = args.batch_size if args.batch_size else 32

# batch ratio for using only batch_ratio % of dataset
batch_ratio = args.batch_ratio if args.batch_ratio else 1

# image standardization
avgs = [float(avg.strip()) for avg in args.avgs.split(',')] \
       if args.avgs else [0.485, 0.456, 0.406]
stdvs = [float(stdv.strip()) for stdv in args.stdvs.split(',')] \
        if args.stdvs else [0.229, 0.224, 0.225]

# image resize
resize = args.resize if args.resize else 256

# image rotation
rot = args.rot if args.rot else 30

# datasets phases
phases = ['train', 'valid', 'test']

# gpu flag
gpu = False if args.gpu == None or args.gpu == 'cpu' else True

# number of epochs
num_epochs = args.epochs if args.epochs else 10

# start learning rate
start_lr = args.start_lr if args.start_lr else 0.001

# end learning rate - if defined by user sheduler will be used
end_lr = args.end_lr if args.end_lr else None

# directory to save model to
save_dir = args.save_dir if args.save_dir else 'checkpoint.pth'

# INITIALIZE MODEL
# ----------------
# initialize selected models for transfer learning
# store model architecture and lr to model id attribute
if end_lr:
    lr = '<{}; {}>'.format(start_lr, end_lr)
else: lr = start_lr
model_id = 'arch={}|lr={}'.format(str(hidden_layers), lr)

# initialize model
net, input_size = model.initialize_model(net_name, num_classes, feature_extract,
                                     pretrained, hidden_layers, drop_p,
                                     model_id)

# LOAD AND PROCESS IMAGES
# -----------------------
image_datasets, dataloaders, dataset_sizes = utils.load_data(
    data_dir, phases, batch_size, rot, resize, input_size, avgs, stdvs )

# TRAIN Model
# -----------
# use gpu for training if requested
device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')

# define loss function
criterion = nn.CrossEntropyLoss()

# define optimization algorithm
optimizer = optim.Adam(model.params_to_update(net.parameters()), lr=start_lr)

# define learning rate scheduler if requested
if end_lr:
    step_size = 1    # number of epochs todecay start_lr by gamma factor
    # lr decay parameter end_lr = start * lr ** (epochs / step_size)
    gamma = (end_lr / start_lr)**(step_size / num_epochs)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = None

# move model to gpu if enabled
net = net.to(device)

# run training
# keep workspace active
with active_session():
    net = model.train_model(model=net, dataloaders = dataloaders,
                                  criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, num_epochs=num_epochs,
                                  batch_ratio = batch_ratio, device=device)

    # SAVE MODEL
    # ----------
    # get mapping of classes numbers/codes to indexes
    class_to_idx = image_datasets[phases[0]].class_to_idx
    model.save_model(net, optimizer, scheduler, class_to_idx, path=save_dir)
    print('Model: {} with id: {} saved to {}.\n'
              .format(net.name, net.id, save_dir))
