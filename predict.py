'''
Module predicts the flower class name from the image using following process
- read arguments
- load neural network model
- process image for prediction
- predict images
'''
# set dependencies
# basics
import numpy as np
import os
import warnings
warnings.simplefilter('ignore')
import time
import argparse

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms

# image processing
from PIL import Image
import json

# custom made modules
import model
import utils

# measure prediction time
start_time = time.time()

# READ ARGUMENTS
# --------------
# DEFINE ARGUMENTS
parser = argparse.ArgumentParser (description = "Parser of prediction script")
parser.add_argument ('image_dir', help = 'Provide path to image. Mandatory argument', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint. Mandatory argument', type = str)
parser.add_argument ('--top_k', help = 'Top K most likely classes. Optional', type = int)
parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)
parser.add_argument ('--gpu', help = "Option to use GPU. Optional", type = str)

# INITIALIZE AND READ ARGUMENTS
args = parser.parse_args()

# read location of model chackpoint
load_dir = args.load_dir

# read device to make prediction on
gpu = False if args.gpu == None or args.gpu == 'cpu' else True

# read location of the image
image_dir = args.image_dir

# read top k lasses to predict
top_k = args.top_k if args.top_k else 1

# read categories names file
class_names_dir = args.category_names if args.category_names else 'cat_to_name.json'
with open(class_names_dir, 'r') as f:
    class_to_name = json.load(f)

# LOAD TRAINED MODEL CHECKPOINT
# -----------------------------
# use gpu for prediction if requested
device = torch.device('cuda:0' if torch.cuda.is_available() and gpu else 'cpu')

net, _, _, = model.load_model(path=load_dir, mode='valid')

# PROCESS IMAGE AND PREDICT IMAGE CLASS
# -------------
# move model to gpu if enabled
net = net.to(device)

# predict image class
top_pbs, top_cls, top_nms = model.predict(image_dir, net, topk=top_k,
                                          device=device,
                                          class_to_name=class_to_name)
# get image class number
image_class_num = image_dir.split('/')[-2]

# get image title
image_class_name = class_to_name[image_class_num]

# print prediction
text =  'Prediction of class: {}'.format(image_class_name)
print('\n'+text)
print('*' * len(text))

for number in range(top_k):
    print('number: {}/{}.. '.format(number+1, top_k),
          'class name: {:25}.. '.format(top_nms[number]),
          'probability: {:.3f}'.format(top_pbs[number]))
# prediction time
elapsed_time = time.time() - start_time
print('\nprediction time: {:.0f}min {:.0f}s\n'
     .format(elapsed_time // 60, elapsed_time % 60))