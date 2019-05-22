'''
Module for utilities for image loading and preprocessing
'''

# IMPORT DEPENDENCIES
# -------------------
# basics
import os
import warnings
warnings.simplefilter('ignore')

# scientific packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# data processing support
import torchvision
from torchvision import datasets, transforms

# image processing
from PIL import Image


def load_data(data_dir, phases, batch_size, rot, resize, input_size, avgs, stdvs):
    '''
    Functions loads and augment images and stores them to dataloaders as input to
    neural network

    Args:
        data_dir: path to root data directory holding images
        phases: phases of image datasets ex. ['train', 'valid' 'test']
        batch_size: size of the images min-batch feeded to network at once
        rot: rotation in degrees for random rotation of images
        resize: shorter size of image in pixls for image resizing
        input_size: size of the image in pixels after image center cropping
        avgs: list of averages per channel used for image standardization
        stdvs: list of standard deviations per chanenl for image standardization

    Returns:
        dataloaders: dictionary of dataloaders per each phase: dataloaders[phase]=dataloader
        dataset_sizes: dictionary of dataset sizes  per each pahse: dataset_sizes[phase]=dataset_size
    '''
    # image processing for each datase phase
    data_transforms = {
        phases[0]: transforms.Compose([
            transforms.RandomRotation(rot),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(avgs, stdvs)
        ]),
        phases[1]: transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(avgs, stdvs)
        ]),
        phases[2]: transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(avgs, stdvs)
        ])}

    # load the datasets with ImageFolder
    image_datasets = {phase: datasets.ImageFolder(
        os.path.join(data_dir, phase), data_transforms[phase])
        for phase in phases}

    # define the dataloaders
    dataloaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase], batch_size=batch_size, shuffle=True)
        for phase in phases}

    # explore datasets sizes
    dataset_sizes = {
        phase: len(image_datasets[phase]) for phase in phases}

    return image_datasets, dataloaders, dataset_sizes

def process_image(image_path, avgs=[0.485, 0.456, 0.406], stdevs=[0.229, 0.224, 0.225],
                  resize=256, crop = 224):
    '''
    Scales (normalizes & standardizes), resize and crops a PIL image for a PyTorch model

    Args:
        image_path: path to image file including name
        resize: number of pixels of shorter size after image resizing
        avgs: array-like object - averages per each image channel to standardize to
        stdev: array-like object - standard deviations per each image channel to standardize to
        crop: crop size to center cropg

    Returns:
        image: processed image as numpy array of shape (Channels, Width, Height)
    '''

    # Process a PIL image for use in a PyTorch model
    # open the image
    img_PIL = Image.open(image_path)

    # RESIZE IMAGE to keep shorter size 256pxl while keeping aspect ratio
    # get image size
    size = np.array(img_PIL.size)

    # calculate aspect ratio
    min_size, max_size = size.argmin(), size.argmax()
    asp_ratio = size[min_size] / size[max_size]

    # resize image
    size[min_size] = int(resize)
    size[max_size] = int(resize) / asp_ratio
    img_PIL = img_PIL.resize(tuple(size))

    # CENTER CROP IMAGE to size 224x224
    # crop margins
    left = int((img_PIL.width - int(crop)) / 2)
    lower = int((img_PIL.height - int(crop)) / 2)
    right= left + int(crop)
    upper = lower + int(crop)

    # crop image
    img_PIL = img_PIL.crop((left, lower, right, upper))

    # SCALE IMAGE
    # normalize to range 0-1
    min_range = 0
    max_range = 255
    img = np.array(img_PIL)
    img = (img - min_range) / (max_range - min_range)

    # standardize
    img = (img - np.array(avgs)) / np.array(stdevs)

    # RESHAPE IMAGE
    # move channels to first dimension to complay with pytorch
    img = img.transpose((2, 0, 1))

    return img
