import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def conv2d(groups =1, dilation=1):
    os.environ['MKLDNN'] = '0'
    input = torch.randn(20, 16, 50, 100)
    #input = torch.randn(1, 1, 10, 10)
    input1 = input.clone()
    input2 = input.clone()
    input1.requires_grad_(True)
    input2.requires_grad_(True)
    conv1= nn.Conv2d(16, 36, (3, 5), stride=(2, 1), padding=(4, 2), groups =groups, dilation=dilation)
    
    #deconv1= nn.ConvTranspose2d(1, 1, (3, 5), stride=(2, 1), padding=(4, 2), groups =groups, dilation=dilation )
    forward1 = conv1(input1)
    #print(foward1)
    y1 = forward1.sum()
    y1.backward()
    input_grad1 = input1.grad.clone()

    # for mkldnn
    os.environ['MKLDNN'] = '1'
    conv2 = copy.deepcopy(conv1)
    forward2 = conv2(input2)
    #print(foward2)
    y2 = forward2.sum()
    y2.backward()
    input_grad2 = input2.grad
    # forward
    #print(foward1)
    #print(foward2)
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

def conv3d(groups =1, dilation=1):
    os.environ['MKLDNN'] = '0'
    input = torch.randn(20, 16, 10, 50, 100)
    #input = torch.randn(1, 2, 2, 5, 2)
    input1 = input.clone()
    input2 = input.clone()
    input1.requires_grad_(True)
    input2.requires_grad_(True)
    conv1= nn.Conv3d(16, 36, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2), groups = groups, dilation=dilation)
    #deconv1= nn.ConvTranspose3d(2, 2, (2,1,2), groups = groups)
    forward1 = conv1(input1)
    #print(foward1)
    y1 = forward1.sum()
    y1.backward()
    input_grad1 = input1.grad.clone()

    # for mkldnn
    os.environ['MKLDNN'] = '1'
    conv2 = copy.deepcopy(conv1)
    forward2 = conv2(input2)
    #print(foward2)
    y2 = forward2.sum()
    y2.backward()
    input_grad2 = input2.grad
    # forward
    #print(foward1)
    #print(foward2)
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

if __name__ == '__main__':
    conv2d()
    # group
    conv2d(groups=2)
    conv2d(groups=4)
    #dilation
    conv2d(dilation=3)
    conv2d(groups=2, dilation=3)
    conv2d(groups=4, dilation=3)

    conv2d(dilation=(3, 1))
    conv2d(groups=2, dilation=(3, 1))
    conv2d(groups=4, dilation=(3, 1))

    # 3d
    print("3d conv")
    conv3d()
    #group
    conv3d(groups=2)
    conv3d(groups=4)

    #dilation
    conv3d(dilation=3)
    conv3d(groups=2, dilation=3)
    conv3d(groups=4, dilation=3)

    conv3d(dilation=(3, 1, 2))
    conv3d(groups=2, dilation=(3, 1, 2))
    conv3d(groups=4, dilation=(3, 1, 2))
