import os
import sys
import json
import torch
import shutil
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.autograd import Variable

# save best model
def save_checkpoint(state, SaveName, epoch, is_best_loss,is_best_kappa, is_best_acc, is_best_jaccard):
    filename = os.path.join("../weights/{}/".format(SaveName),"checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename, os.path.join("../weights/{}/".format(SaveName), "loss.pth.tar"))
    if is_best_kappa:
        shutil.copyfile(filename, os.path.join("../weights/{}/".format(SaveName), "kappa.pth.tar"))
    if is_best_acc:
        shutil.copyfile(filename, os.path.join("../weights/{}/".format(SaveName), "acc.pth.tar"))
    if is_best_jaccard:
        shutil.copyfile(filename, os.path.join("../weights/{}/".format(SaveName), "jaccard.pth.tar"))


# evaluate meters
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

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%dhr%dmin'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        hr = (t//60)//60
        min = t//60
        sec = t%60
        return '%2d:%2d:%02d'%(hr,min,sec)

    elif mode=='int':
        return '%03ds'%(t)
    else:
        raise NotImplementedError
