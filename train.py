# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import joblib
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import torch.backends.cudnn as cudnn
from sklearn.metrics import precision_recall_fscore_support
from dataset4 import Dataset as Dataset
import collections
from fcn5 import FCN 
from utils import str2bool, count_params


#arch_names = list(archs.__dict__.keys())
#arch_names = ''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    parser.add_argument('--log', default=None,
                        help='log File name')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=None, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    model.train()
    #accuracy = AverageMeter()
    acc_mean = 0
    count = 0
    acc = 0
    gt = []
    pr = []
    pr2 = []
    for i, (id,input, target) in enumerate(train_loader): #, total=len(train_loader):
        acc = 0
        input = input.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        # compute output
        output = model(input)
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        
        output_ = output.detach().cpu().numpy() 
        target_ = target.detach().cpu().numpy()
        #output2 = torch.softmax(output.detach().cpu(),1).numpy()
        output2 = output_
        output_ = output_.argmax(1)
        #target_ = target_.argmax(1)
        
        acc = output_ == target_
        acc = acc.sum().item()
        count += len(output_)
        acc_mean += acc
        # compute gradient and do optimizing step
        
        loss.backward()
        optimizer.step()
        for i in range(0,len(target_)):
            gt.append(target_[i])
            pr.append(output_[i])
            pr2.append(output2[i][1])
    
    acc_mean = acc_mean/float(count)
    fpr,tpr,thresholds_keras = roc_curve(gt,pr2)
    auroc = auc(fpr,tpr)
    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pr, pos_label=1,average='binary')
    log = OrderedDict([
        ('loss', losses.avg), ('acc', acc_mean), ('precision',precision), ('recall', recall), ('f1', f_score),('auroc',auroc)
    ])

    return log

def validate(args, valid_loader, model, criterion):
    losses = AverageMeter()
    model.eval()
    acc_mean = 0
    count = 0
    acc = 0
    gt = []
    pr = []
    pr2 = []
    with torch.no_grad():
        for i, (id,input, target) in enumerate(valid_loader): #, total=len(train_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            
            output_ = output.cpu().numpy() 
            target_ = target.cpu().numpy()
            #output2 = torch.softmax(output.cpu(),1).numpy()
            output2 = output_
            output_ = output_.argmax(1)
            #target_ = target_.argmax(1)
            acc = output_ == target_
            acc = acc.sum().item()
            count += len(output_)
            acc_mean += acc
            for i in range(0,len(target_)):
                gt.append(target_[i])
                pr.append(output_[i])
                pr2.append(output2[i][1])
        fpr,tpr,thresholds_keras = roc_curve(gt,pr2)
        auroc = auc(fpr,tpr)
        acc_mean = acc_mean/float(count)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pr, pos_label=1, average='binary')
       
    log = OrderedDict([
        ('loss', losses.avg), ('acc', acc_mean), ('auroc',auroc),('precision',precision), ('recall', recall), ('f1', f_score),('auroc',auroc)
    ])

    return log



def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.name is None:
       args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    #criterion = losses.__dict__[args.loss]().cuda()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.SmoothL1Loss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = torch.nn.BCELoss()
    
    cudnn.benchmark = True

    # Data loading code
   
    datasetPath = args.dataset 
    
    
    # create model
    #print("=> creating model %s" %args.arch)
    #model = archs.__dict__[args.arch](args)
    #model = FCN()
    #model = nn.DataParallel(model)
    #model.to(device)
    #model = model.cuda()

    #print(count_params(model))

    #if args.optimizer == 'Adam':
    #    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    #elif args.optimizer == 'SGD':
    #    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    #        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
   
    #X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2, random_state=0)


    #train_dataset = Dataset(trainPatDateList, train_target)
    #val_dataset = Dataset(valPatDateList, val_target)
    dataset = Dataset(datasetPath)
    
    if dataset.posNum > dataset.negNum:
        weight = [dataset.posNum/dataset.negNum, 1.0]
    else:
        weight = [1.0,dataset.negNum/dataset.posNum]
    weight = torch.FloatTensor(weight)
    weight = weight.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    k = 5
    splits=KFold(n_splits=k, shuffle=True, random_state=720319)    
    foldperf={}
    logFile = open(os.path.join('models', args.name ,args.log) ,'w')
    modelList = []
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))
        logFile.write('Fold {}'.format(fold + 1) + '\n')
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = FCN()
        #model = nn.DataParallel(model)
        model.to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        #model = ConvNet()
        #model.to(device)
        #optimizer = optim.Adam(model.parameters(), lr=0.002)

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        
        for epoch in range(args.epochs):
            #print(epoch)
            train_log = train(args,train_loader,model,criterion,optimizer)
            test_log = validate(args,val_loader,model,criterion)
            torch.save(model, os.path.join('models',args.name,str(fold+1) + '_fold_' + str(epoch+1) + '.pt'))
            #train_loss = train_loss / len(train_loader.sampler)
            #train_acc = train_correct / len(train_loader.sampler) * 100
            #test_loss = test_loss / len(val_loader.sampler)
            #test_acc = test_correct / len(val_loader.sampler) * 100
            logFile.write("Epoch:{}/{} Training\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n             Testing\t{:.3f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(epoch + 1,
                                                                                                                args.epochs,
                                                                                                                train_log['loss'],
                                                                                                                train_log['acc'],
                                                                                                                train_log['precision'],
                                                                                                                train_log['recall'],
                                                                                                                train_log['f1'],
                                                                                                                train_log['auroc'],
                                                                                                                test_log['loss'],
                                                                                                                test_log['acc'],
                                                                                                                test_log['precision'],
                                                                                                                test_log['recall'],
                                                                                                                test_log['f1'],
                                                                                                                test_log['auroc']
                                                                                                               ))
            print("Epoch:{}/{} Training:{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\n                Testing:{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(epoch + 1,
                                                                                                                args.epochs,
                                                                                                                train_log['acc'],
                                                                                                                train_log['precision'],
                                                                                                                train_log['recall'],
                                                                                                                train_log['f1'],
                                                                                                                train_log['auroc'],
                                                                                                                test_log['acc'],
                                                                                                                test_log['precision'],
                                                                                                                test_log['recall'],
                                                                                                                test_log['f1'],
                                                                                                                test_log['auroc']
                                                                                                               ))
            
            
                                                                                                              
            history['train_loss'].append(train_log['loss'])
            history['test_loss'].append(test_log['loss'])
            history['train_acc'].append(train_log['acc'])
            history['test_acc'].append(test_log['acc'])
            torch.cuda.empty_cache()
            
        
        #modelList.append(os.path.join('models',args.name,str(fold+1) + '_fold.pt'))
        #foldperf['fold{}'.format(fold+1)] = history  
    '''
    state_dictList = [torch.load(x, map_location='cpu').state_dict() if x.endswith('pt') else torch.load(x, map_location='cpu')['state_dict'] for x in modelList]
    ave_state_dict = collections.OrderedDict()
    weight_keys=list(state_dictList[0].keys())
    
    for key in weight_keys:
        key_sum = 0
        for i in range(len(modelList)):
            key_sum += state_dictList[i][key]
        ave_state_dict[key] = key_sum/float(len(modelList))
    model.load_state_dict(ave_state_dict)
    torch.save(model,os.path.join('models',args.name, 'ave' + str(fold+1) + '_fold.pt'))
    '''
    #torch.save(model,'k_cross_fcn_with_density.pt')    


        


if __name__ == '__main__':
    main()
