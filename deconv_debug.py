import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def deconv2d(groups =1, dilation=1):
    os.environ['MKLDNN'] = '0'
    input = torch.randn(20, 16, 50, 100)
    #input = torch.randn(1, 1, 10, 10)
    input1 = input.clone()
    input2 = input.clone()
    input1.requires_grad_(True)
    input2.requires_grad_(True)
    deconv1= nn.ConvTranspose2d(16, 36, (3, 5), stride=(2, 1), padding=(4, 2), groups =groups, dilation=dilation)
    
    #deconv1= nn.ConvTranspose2d(1, 1, (3, 5), stride=(2, 1), padding=(4, 2), groups =groups, dilation=dilation )
    forward1 = deconv1(input1)
    #print(foward1)
    y1 = forward1.sum()
    y1.backward()
    input_grad1 = input1.grad.clone()

    # for mkldnn
    os.environ['MKLDNN'] = '1'
    deconv2 = copy.deepcopy(deconv1)
    forward2 = deconv2(input2)
    #print(foward2)
    y2 = forward2.sum()
    y2.backward()
    input_grad2 = input2.grad
    # forward
    #print(forward1)
    #print(forward2)
    print((forward1-forward2).abs().max())
    if (forward1-forward2).abs().max()<1e-5:
       print("the forward is same in forward")
    else :
       print("the forward is not same in training")
    # backward
    print((input_grad1-input_grad2).abs().max())
    if (input_grad1-input_grad2).abs().max()<1e-5:
       print("the backward is same in training")
    else:
       print("the backward is not same in training")

def deconv3d(groups =1, dilation=1):
    os.environ['MKLDNN'] = '0'
    input = torch.randn(20, 16, 10, 50, 100)
    #input = torch.randn(1, 2, 2, 5, 2)
    input1 = input.clone()
    input2 = input.clone()
    input1.requires_grad_(True)
    input2.requires_grad_(True)
    deconv1= nn.ConvTranspose3d(16, 34, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2), groups = groups, dilation=dilation)
    #deconv1= nn.ConvTranspose3d(2, 2, (2,1,2), groups = groups)
    foward1 = deconv1(input1)
    #print(foward1)
    y1 = foward1.sum()
    y1.backward()
    input_grad1 = input1.grad.clone()

    # for mkldnn
    os.environ['MKLDNN'] = '1'
    deconv2 = copy.deepcopy(deconv1)
    foward2 = deconv2(input2)
    #print(foward2)
    y2 = foward2.sum()
    y2.backward()
    input_grad2 = input2.grad
    # forward
    #print(foward1)
    #print(foward2)
    print((foward1-foward2).abs().max())
    if (foward1-foward2).abs().max()<1e-6:
       print("the forward is same in forward")
    else :
       print("the forward is not same in training")
    # backward
    if (input_grad1-input_grad2).abs().max()<1e-6:
       print("the backward is same in training")
    else:
       print("the backward is not same in training")

if __name__ == '__main__':
    deconv2d()
    #deconv2d(groups=2)
    #deconv3d()
    #deconv3d(groups=2)
    #deconv2d(dilation=2)
    #deconv3d(dilation=2)
