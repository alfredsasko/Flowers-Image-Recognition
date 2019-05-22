'''
Module for classes and functions related to neural network model
used for transfer learning
'''
# IMPORT DEPENDENCIES
# -------------------
# basics
import time
import os
import copy
import warnings
warnings.simplefilter('ignore')

# scientific packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# data processing support
import torchvision
from torchvision import models

# custom made Modules
import utils

class FcNetwork(nn.Module):
    '''
    Builds a feedforward network with arbitrary hidden layers
    with ReLU activation function and drop out

    Note: Forward method returns logits (raw outputs)

    Args:
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, the sizes of the hidden layers
    '''
    def __init__(self, input_size, output_size, hidden_layers=None, drop_p=0.5):

        super().__init__()

        # drop out probability attribute

        # if hidden leayers are not requested
        if (hidden_layers == None) or (hidden_layers == []):
            self.drop_p = None
            self.hidden_layers = None
            self.output = nn.Linear(input_size, output_size)
        else:
            # hidden layers requested
            # attribute drop out probability
            self.drop_p = drop_p

            # 1st hidden layer
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

            # add a variable number of  more hidden layers
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(in_features, out_features)
                                      for in_features, out_features in layer_sizes])

            # regularization
            self.dropout = nn.Dropout(p=self.drop_p)

            # output of the network
            self.output = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, x):
        '''
        Forward pass trough network ,returns the ouput logits (raw outputs)
        '''
        # forward input trough hidden layers if they exists
        if not (self.hidden_layers == None):
            for layer in self.hidden_layers:
                x = layer(x)
                x = F.relu(x)
                x = self.dropout(x)

        # forward input trough last layer
        output = self.output(x)

        return output

    def layer_sizes(self):
        '''
        Returns model architecture as list of layer sizes in string format.
        Example of layer with 25k inputs, 2048 nodes of 1st hidden layer,
        1024 of sencond, 512 of third and 102 outputs is equal to string
        [25000, 2048, 1024, 512, 102]

        Returns:
            architecture: string as list of neural network layer layer_sizes
        '''

        layer_sizes = []
        #  no hidden layers defined
        if self.hidden_layers == None:
            # gent # of inputs to output layer
            layer_sizes.append(self.output.in_features)
        #  hidden leayers defined
        else:
            # inputs to first hidden layer
            layer_sizes.append(self.hidden_layers[0].in_features)
            # cycle over hidden layers
            for hidden_layer in self.hidden_layers:
                # get hidden layer nodes
                layer_sizes.append(hidden_layer.out_features)

        # add # of outputs | classes | nodes of output layer
        layer_sizes.append(self.output.out_features)

        return str(layer_sizes)

def freeze_model_params(model, feature_extract=True):
    '''
    Freeze model parameters for feature extraction type of transfer learning

    Args:
        model: neural network model of nn.Module class or its child
        feature_extract: transfer learning type feature extraction ro
                         tunning, deafult: feature_extract=True
    '''
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True,
                     pretrained=True, hidden_layers=None, drop_p=0.5, model_id=None, class_to_idx=None):
    '''
    Initialize selected model for transfer learning

    note: following list of models is supported [vgg, resnet]
          and can be further extended

    Args:
        model_name: name of the model
        num_classes: number of classes in the output of designed network

        feature_extract: transfer learning type feature extraction or
                         tunning, default: feature_extract=True

        pretrained: network weight initialization: pretrained or
                    not pretrained, dafault: use_pretrained=True

        hidden_layers: classifier architecture need to be fully connected network (n+2 layers)
                              with drop out, defined as list of sizes of hidden layers:
                              [num_hidden_layer_1, num_hidden_layer_2, ..., num_hidden_layer_n]
                              example: [2048, 1024, 512]

                              Note: input layer is defined by model name and output by num_classes

        drop_p: drop out probability

    Returns:
        model: model adapted for transfer learning with initialized weights
        input_size: required input_size of the model
    '''

    # initialize returns
    model = None
    input_size = 0
    if model_id == None: model_id = model_name

    if 'vgg' in model_name:
        # initialize model
        model = models.vgg11_bn(pretrained=pretrained)

        # freeze model parameters in case of transfer learning type: feature extraction
        freeze_model_params(model, feature_extract)

        # get number of input nodes of the classifier laeyer to replace
        in_features = model.classifier[0].in_features

        # rebuild model and remove last layer of the classifier
        model.classifier = FcNetwork(in_features, num_classes, hidden_layers, drop_p=drop_p)

        # create attribute model name & id
        model.name = model_name
        model.id = model_id

        # create attribute storking mappings of class codes to indexes
        model.class_to_idx = class_to_idx

        # define number of inputs the model was trained on
        input_size = 224

    elif 'resnet' in model_name:
        model = models.resnet50(pretrained=pretrained)
        freeze_model_params(model, feature_extract)
        in_features = model.fc.in_features
        model.fc = FcNetwork(in_features, num_classes, hidden_layers, drop_p=drop_p)
        model.name = model_name
        model.id = model_id
        model.class_to_idx = class_to_idx
        input_size = 224

    else:
        print('Invalid model name.')

    return model, input_size

def params_to_update(model_params):
    '''
    Returns model parameters which will be tuned during training

    Arg:
        model_params: neural network model parameters as generator
                      Note: use nn.Module.parameters()
    Returns:
        generator of parameters to be updated during training
    '''

    for param in model_params:
        if param.requires_grad == True:
            yield param

def train_model(model, dataloaders, criterion, optimizer, phases=['train', 'valid'],
                scheduler=None, num_epochs=10, batch_ratio=1, device='cpu'):
    '''
    Train neural network with arbitrary architecture, loss function, optimization algorithm,
    adaptive learning rate and returns model with best accuracy

    Args:
        model: model with arbitrary architecture of torch.nn module
        dataloaders: dictionary of data loaders per learning phase,
                     dataloader interates in batches trough data in each phase
        criterion: loss fuction of torch.nn module
        optimizer: optimization algorithm of torch.optim module
        phases: list of learning phases one or some of ['train', 'valid']
        sheduler: adpative learning rate sheduler of torch.optim module
        num_epochs: number of epochs to train model on, default is 15
        batch_ratio: iterates over bath_ration % of batches in dataset
        device: device to train model on: default cpu or gpu

    Returns:
        model with updated weights corresponding to highest accuracy
    '''

    # debugging on batch results will be printed
    debug_mode = False

    # frequency of printing results every x batch
    print_every = 1

    # track training time
    since = time.time()

    # store best model weights
    best_model_wts = copy.deepcopy(model.state_dict())

    # initiate accuracy tracking
    best_acc = 0.0

    # print model name
    print_model = '\nmodel: {} ---> id: {}'.format(model.name, model.id)
    print(print_model)
    print('*' * len(print_model), '\n')

    # model training and validation
    for epoch in range(num_epochs):
        # track epoch running time
        epoch_since = time.time()

        # notify user on running epoch
        print_epoch = 'epoch {}/{}'.format(epoch + 1, num_epochs)
        print(print_epoch)
        print('-' * len(print_epoch))

        # Cycles trough epoch training and validation phase
        for phase in phases:
            # monitor phase time
            phase_since = time.time()

            if phase == 'train':
                # update learning rate if using sheduler
                if not (scheduler == None):
                    scheduler.step()
                model.train()    # set model in training mode
            else:
                model.eval()    # set model in evaluation mode

            # reset loss, corrects and running time from previous epoch
            running_loss = 0.0
            running_corrects = 0
            running_time = 0

            # reset running number of samples
            running_size = 0

            # iterate over n batches in dataset
            num_batches = max(1, int(len(dataloaders[phase]) * batch_ratio))
            for batch, (inputs, labels) in zip(range(num_batches), dataloaders[phase]):

                # trace batch time in debug mode
                if debug_mode: batch_since = time.time()

                # update number of samples
                # import ipdb; ipdb.set_trace()
                running_size += inputs.size(0)

                # transfer data to gpu if enabled
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients from previous batch
                optimizer.zero_grad()

                # track operations on model parameters to calculate gradient
                # only for training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # calculate model output
                    outputs = model(inputs)

                    # calculate loss
                    loss = criterion(outputs, labels)

                    # calculate top k prediction
                    _, preds = outputs.topk(1, dim=1)

                    # backward step, calculate and update weights in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # accumulate loss, number of correct clasifications and running time
                # in batch and phase
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.view(preds.shape))

                # with debug on, print batch results
                if debug_mode and (phase == 'train') and ((batch + 1) % print_every == 0) :
                    # calculate batch time
                    batch_time = time.time() - batch_since

                    # calculate average running batch loss and accuracy for phase
                    batch_loss = loss.item()
                    batch_acc = torch.sum(preds == labels.view(preds.shape)).double() / inputs.size(0)

                    # print batch results per phase
                    print('{} batch {:3}/{:3} loss: {:.4f} Acc: {:.4f} Time:{:.0f}s'
                         .format(phase, batch+1, num_batches,
                                batch_loss, batch_acc, batch_time))

            # calculate epoch loss and accuracy for phase
            epoch_phase_loss = running_loss / running_size
            epoch_phase_acc = (running_corrects.double() / running_size).item()
            epoch_phase_time = time.time() - phase_since


            # print epoch results per phase
            print('{} loss: {:.4f} acc: {:.4f} time: {:.0f}m {:.0f}s'
                  .format(phase, epoch_phase_loss, epoch_phase_acc,
                         epoch_phase_time // 60, epoch_phase_time % 60))

            # store weights if validation accuracy is best
            if phase == 'valid' and epoch_phase_acc > best_acc:
                best_acc = epoch_phase_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # ad space after epoch results
        print()

    # record and print training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # print best validation accuracy
    print('Best val Acc: {:.4f}\n'.format(best_acc))

    return model

def save_model(model, optimizer=None, scheduler=None, class_to_idx=None, path='checkpoint.pth'):
    '''
    Save pytorch model for inference and/or resuming training.

    Args:
        model: any instance of pytorch.models where last layer is adapted
               using FcNetwork class

        optimizer: any optimization algorithm of pytorch.optim class,
                   use only for resuming training purpose, default None
        sheduler: any learning rate sheduler nn.optim.lr_sheduler class
                  us enly for resuming training purpose if it was used,
                  defulat None
        class_to_idx: dictionary mapping classes numbers/codes to indexes
        path: path and file name, default 'checkpoint.pth'

    Returns:
        saved model file on defined path location
    '''
    # reference to classifier (last) module of the model
    classifier = list(model.children())[-1]

    # save best model
    torch.save({'model':
                {'id': model.id,
                 'name': model.name,
                 'num_classes': classifier.output.out_features,
                 'feature_extract': ~np.all([param.requires_grad for param in model.parameters()]),
                 'hidden_layers': None if classifier.hidden_layers == None else
                                  [layer.out_features for layer in classifier.hidden_layers],
                 'dropout_prob': classifier.drop_p,
                 'state_dict': model.state_dict(),
                 'class_to_idx': class_to_idx
                },
                'optimizer': None if optimizer == None else{
                    'state_dict': optimizer.state_dict()},
                'lr_scheduler': None if scheduler == None else {
                    'step_size': scheduler.step_size,
                    'state_dict': scheduler.state_dict()},
               },
               path
              )

def load_model(path='checkpoint.pth', mode='valid', pretrained=True):
    '''
        Load Model for inference or further training

        Args:
            mode: load mode: inference = 'valid', default: training = 'train'
            pretrained: True if we want to keep trained parameters, False to
                        initialize them randomly

        Returns:
            model: intialized model of class torch.models
            optimizer: initialized optimizer used for tranining
            sheduler: initialized learning rate sheduler if it was used for training
            history: history of the last training

        Note: last tree items are only returned in traning mode
    '''

    # load checkpoint
    checkpoint = torch.load(path)

    # initialize model
    model, _ = initialize_model(
        checkpoint['model']['name'],
        checkpoint['model']['num_classes'],
        checkpoint['model']['feature_extract'],
        pretrained,
        checkpoint['model']['hidden_layers'],
        checkpoint['model']['dropout_prob'],
        checkpoint['model']['id'])

    # restore model parameters
    model.load_state_dict(checkpoint['model']['state_dict'])

    # restore mapping of class codes to idexes
    model.class_to_idx = checkpoint['model']['class_to_idx']

    # initialize optimizer
    if mode == 'valid' or checkpoint['optimizer'] == None:
        optimizer = None
    else:
        optimizer = optim.Adam(params_to_update(model.parameters()))
        # restore optimizer parameters
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])

    # initialize lr scheduler
    if mode == 'valid' or checkpoint['lr_scheduler'] == None:
        scheduler = None
    else:
        scheduler =  lr_scheduler.StepLR(optimizer, checkpoint['lr_scheduler']['step_size'])
        # restore scheduler parameters
        scheduler.load_state_dict(checkpoint['lr_scheduler']['state_dict'])

    # set model to training or inference mode
    model.train() if mode == 'train' else model.eval()

    return model, optimizer, scheduler

def predict(image_path, model, class_to_name, topk=1, device='cpu'):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.

    Args:
        image_path: path to the image which class need to be predicted
        model: neural network model of type nn.Module
        topk: number of top k predictions, integer
        device: device to predict on cpu by default or gpu
        class_to_name: dictionary mapping classes numbers/codes to names

    Returns:
        top_pbs: list of top k probabilities
        top_cls: list of top k classes numbers
        top_nms: list of top k classes names
    '''

    # IMAGE PROCESSING
    # scale, resize and crop image, return numpy array
    img = utils.process_image(image_path)

    # transform image to tensor of shape (batch_size, channel, width, height)
    img = (torch.from_numpy(img)      # create tensor from numpy
           .type(torch.FloatTensor)   # recast tensor to float dtype
           .unsqueeze(0))             # add 1 dimension left for batches

    # PREDICTION
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking to speed up prediction
    with torch.set_grad_enabled(False):
        # calculate raw outputs
        logits = model(img.to(device))

        # calculate probabilities
        pbs = nn.Softmax(dim=1)(logits)

        # select top k probabilities and classes indexes
        top_pbs, top_idx = pbs.topk(topk)

        # transform probabilities and classes idexes to list
        top_pbs = top_pbs.cpu().detach().numpy().tolist()[0]
        top_idx = top_idx.cpu().detach().numpy().tolist()[0]

    # RETURN probabilities, classes and classes names as list
    idx_to_class = dict(map(reversed, model.class_to_idx.items()))
    top_cls = [idx_to_class[idx] for idx in top_idx]
    top_nms = [class_to_name[top_cl] for top_cl in top_cls]

    return top_pbs, top_cls, top_nms
