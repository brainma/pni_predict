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
from dataset4 import inferDataset as Dataset

from fcn5 import FCN 
from utils import str2bool, count_params


#arch_names = list(archs.__dict__.keys())
#arch_names = ''


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    
    
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

def validate(args, valid_loader, model, criterion):
    losses = AverageMeter()
    model.eval()
    acc_mean = 0
    count = 0
    acc = 0
    gt = []
    pr = []
    pr2 = []
    idList = []
    with torch.no_grad():
        for i, (id,input, target,pni_value) in enumerate(valid_loader): #, total=len(train_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            output2 = torch.softmax(output.cpu(),1).numpy()
            
            output_ = output.cpu().numpy() 
            target_ = target.cpu().numpy()
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
                idList.append(str(id[i]) + ','+ pni_value[i] +',' + str(target_[i]) + ',' + str(output_[i]))
        #print(pr)
        #print(idList)
        fpr,tpr,thresholds_keras = roc_curve(gt,pr2)
        auroc = auc(fpr,tpr)
        acc_mean = acc_mean/float(count)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pr, average='binary')
       
    log = OrderedDict([
        ('loss', losses.avg), ('acc', acc_mean),('precision',precision), ('recall', recall), ('f1', f_score),('auroc',auroc)
    ])

    return log, idList



def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    
    joblib.dump(args, 'models/%s/args.pkl' %args.model)

    # define loss function (criterion)
    #criterion = losses.__dict__[args.loss]().cuda()
    #criterion = torch.nn.MSELoss()
    #criterion = torch.nn.SmoothL1Loss()
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    cudnn.benchmark = True

    # Data loading code
   
    datasetPath = args.dataset 
    dataset = Dataset(datasetPath)
    foldperf={}
    #logFile = open('test_log.txt','w')
    val_loader = DataLoader(dataset, batch_size=args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FCN()
    #model = nn.DataParallel(model)
    model.to(device)
    epoch_array = [45,46,50,50,43]
    for k in range(5):
        model = torch.load(os.path.join('models',args.model, str(k+1)+'_fold_'+ str(epoch_array[k]) +'.pt') , map_location=device)
        test_log, inferResult = validate(args,val_loader,model,criterion)
    
        output = open(os.path.join('output',args.model + '_' + str(k+1)+'_fold_'+ str(epoch_array[k])+'_118.txt'),'w')
        output.write('ID_DATE,PNI_VALUE,PNI_BINARY,PNI_PREDICTION\n')
        for item in inferResult:
            output.write(item + '\n')
        
        output.close()
        '''
        logFile.write("Testing: Loss {:.3f}, Acc {:.2f}, Precision {:.2f}, Recall {:.2f}, F1 {:.2f}, AUROC {:.2f}\n".format(
                                                                                                                test_log['loss'],
                                                                                                                test_log['acc'],
                                                                                                                test_log['precision'],
                                                                                                                test_log['recall'],
                                                                                                                test_log['f1'],
                                                                                                                test_log['auroc']
                                                                                                               ))        
        
        print("Testing: Loss {:.3f}, Acc {:.5f}, Precision {:.5f}, Recall {:.5f}, F1 {:.5f}, AUROC {:.5f}".format(
                                                                                                                test_log['loss'],
                                                                                                                test_log['acc'],
                                                                                                                test_log['precision'],
                                                                                                                test_log['recall'],
                                                                                                                test_log['f1'],
                                                                                                                test_log['auroc']
                                                                                                               ))
        '''
        print("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(test_log['acc'],test_log['precision'],test_log['recall'],test_log['f1'],test_log['auroc']
                                                                                                               ))
        torch.cuda.empty_cache()
    
    


        


if __name__ == '__main__':
    main()
