import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def pooling(kernel_size, stride=None, padding=0, ceil_mode=True, dim =2):
   
    os.environ['MKLDNN'] = '0'
    if dim ==2:
       input = torch.randn(20, 16, 50, 32)
       #input =  torch.tensor([[[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0.0]]]])
       pool1 = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding, ceil_mode= ceil_mode,)
    else:
       input = torch.randn(20, 16, 50, 44, 31)
       pool1 = nn.MaxPool3d(kernel_size = kernel_size, stride = stride, padding = padding, ceil_mode= ceil_mode,)

    input2=input.clone()
    input.requires_grad_(True)
    input2.requires_grad_(True)
    forward1=pool1(input)
    y1 = forward1.sum()
    y1.backward()
    x_grad1 = input.grad
    # for mkldnn
    print("<<<<<<using mkldnn>>>>>>")
    os.environ['MKLDNN'] = '1'
    pool2 = copy.deepcopy(pool1)
    forward2=pool2(input2)
    y2 = forward2.sum()
    y2.backward()
    x_grad2 = input2.grad
    # forward
    if torch.equal(forward1,forward2):
       print("the forward is same in traning")
    else :
      print("the forward is not same in training")
    # backward
    if torch.equal(x_grad1,x_grad2):
      print("the backward is same in training")
    else:
      print("the backward is not same in training")


if __name__ == '__main__':
    # for max_pool2d
    # pool of non-square window
    
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = 1

    pooling(kernel_size, dim = 2) # ceil model model, 
    pooling(kernel_size, ceil_mode=False, dim = 2) # floor model model, 

    pooling(kernel_size, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=False, dim = 2) # floor model model, 
   

    #pool of square window of size=3, stride=2
    kernel_size = 3
    stride = 2

    pooling(kernel_size, stride, dim = 2) # ceil model model, 
    
    pooling(kernel_size, stride, ceil_mode=False, dim = 2) # floor model model, 

    pooling(kernel_size,stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size,stride, padding=padding, ceil_mode=False, dim = 2) # floor model model, 

    #non square padding
    padding = (1,0)

    pooling(kernel_size,stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size,stride, padding=padding, ceil_mode=False, dim = 2) # floor model model, 
 

    # for max_pool3d
    # pool of non-square window
    kernel_size = (3, 2, 2)
    stride = (2, 1, 2)
    padding = 1

    pooling(kernel_size, dim = 3) # ceil model model, 
    pooling(kernel_size, ceil_mode=False, dim = 3) # floor model model, 

    pooling(kernel_size, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=False, dim = 3) # floor model model, 


    #pool of square window of size=3, stride=2
    kernel_size = 3
    stride=2

    pooling(kernel_size, stride, dim = 3) # ceil model model, 
    pooling(kernel_size, stride, ceil_mode=False, dim =3) # floor model model, 

    pooling(kernel_size,stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size,stride, padding=padding, ceil_mode=False, dim = 3) # floor model model, 
 
    padding = (1,0,1)
    pooling(kernel_size,stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size,stride, padding=padding, ceil_mode=False, dim = 3) # floor model model, 

