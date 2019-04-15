import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def batchnorm(input, num_features, eps = 1e-05, momentum = 0.1,
     affine = True, track_running_stats = True, dim = 2):
    os.environ['MKLDNN'] = '0'

    if dim ==2 :
       input = torch.randn(20, 100, 35, 45)
       batchnorm1 = nn.BatchNorm2d(num_features, eps= eps, momentum = momentum,
           affine=affine, track_running_stats = track_running_stats)
    else:
       input = torch.randn(20, 100, 35, 45, 10)
       batchnorm1 = nn.BatchNorm3d(num_features, eps= eps, momentum = momentum,
           affine=affine, track_running_stats = track_running_stats)
    input2=input.clone()
    input.requires_grad_(True)
    input2.requires_grad_(True)
    forward1=batchnorm1(input)
    y1 = forward1.sum()
    y1.backward()
    input_grad1 = input.grad.clone()
    # for mkldnn
    os.environ['MKLDNN'] = '1'
    batchnorm2 = copy.deepcopy(batchnorm1)
    forward2 = batchnorm2(input2)
    y2 = forward2.sum()
    y2.backward()
    input_grad2 = input2.grad.clone()
    
    print((forward1-forward2).abs().max())
    if (forward1-forward2).abs().max()<1e-5:
       print("the forward is same in traning")
    else :
       print("the forward is not same in training")
    # backward
    if (input_grad1-input_grad2).abs().max()<1e-5:
       print("the backward is same in training")
    else:
       print("the backward is not same in training")

if __name__ == '__main__':
    # for batchnorm2d
    num_features = 100
    batchnorm(input, num_features, eps = 1e-05, momentum = 0.1,
     affine = True, track_running_stats = True, dim = 2 ) # affine = True, track_running_stats = True

    batchnorm(input, num_features, eps = 1e-05, momentum = 0.1,
     affine = True, track_running_stats = False, dim = 2 ) # affine = True, track_running_stats = False

    # for batchnorm3d
    num_features = 100
    batchnorm(input, num_features, eps = 1e-05, momentum = 0.1,
     affine = True, track_running_stats = True, dim = 3 ) # affine = True, track_running_stats = True

    batchnorm(input, num_features, eps = 1e-05, momentum = 0.1,
     affine = True, track_running_stats = False, dim = 3 ) # affine = True, track_running_stats = False
