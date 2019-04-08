import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def Activation(inplace =False, dim =2):
    os.environ['MKLDNN'] = '0'
    input = torch.randn(20, 16, 50, 32)
    if dim == 3:
        input = torch.randn(20, 16, 50, 44, 31)
    input.requires_grad_(True)
    relu1 = nn.ReLU(inplace = inplace)
    input1 = input.clone()
    input2 = input.clone()
    forward1 = relu1(input1)
    y1 = forward1.sum()
    y1.backward()
    input_grad1 = input.grad

    # for mkldnn
    os.environ['MKLDNN'] = '1'
    relu2 = copy.deepcopy(relu1)
    forward2 = relu2(input2)
    y2 = forward2.sum()
    y2.backward()
    if inplace:
       input_grad2 = input.grad
    else:
       input_grad2 = input.grad
    # forward
    if torch.equal(forward1,forward2):
       print("the forward is same in forward")
    else :
       print("the forward is not same in training")
    # backward
    if torch.equal(input_grad1, input_grad2):
       print("the backward is same in training")
    else:
       print("the backward is not same in training")


if __name__ == '__main__':
    # for relu2d
    Activation(inplace = False)
    Activation(inplace = True) # inplace
    # for relu3d
    Activation(inplace = False,dim=3)
    Activation(inplace = True, dim=3) # inplace
