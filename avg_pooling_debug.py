import torch
import torch.nn as nn
import os
import copy
from torch.autograd import Variable

def pooling(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad =True, dim =2):
   
    os.environ['MKLDNN'] = '0'
    if dim ==2:
       input = torch.randn(20, 16, 50, 32)
       pool1 = nn.AvgPool2d(kernel_size = kernel_size, stride = stride, padding = padding, ceil_mode= ceil_mode, count_include_pad=count_include_pad)
    else:
       input = torch.randn(20, 16, 50, 44, 31)
       #input = torch.randn(1, 1, 3, 5, 6)
       pool1 = nn.AvgPool3d(kernel_size = kernel_size, stride = stride, padding = padding, ceil_mode= ceil_mode, count_include_pad=count_include_pad)

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
    #print(forward1)
    #print(forward2)
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
    # for avg_pool2d
    # pool of non-square window
    kernel_size = (3, 2)
    stride = (2, 1)
    padding = 1
 
    # stride = kernel, padding=0
    pooling(kernel_size, dim = 2) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad


    #pool of square window of size=3
    kernel_size = 3
   # stride = kernel, padding=0
    pooling(kernel_size, dim = 2) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
  
    # square stride 
    # stride =2
    pooling(kernel_size, dim = 2) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad

    #non square padding
    padding = (1,0)

    pooling(kernel_size, dim = 2) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 2) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 2) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 2) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 2) # ceil model model, none count_include_pad


    # for avg_pool3d
    # pool of non-square window

    kernel_size = (3, 2, 3)
    stride = (2, 1, 2)
    padding = 1

    # stride = kernel, padding=0
    pooling(kernel_size, dim = 3) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0

    #pooling(kernel_size, stride= stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 3) # floor model model, 

    #pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    #pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
 
    #pool of square window of size=3
    kernel_size = 3
   # stride = kernel, padding=0
    pooling(kernel_size, dim = 3) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
  
    # square stride 
    # stride =2
    pooling(kernel_size, dim = 3) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim =3) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim =3) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad

    #non square padding
    padding = (1,0,1)

    pooling(kernel_size, dim = 3) # ceil model model, 
    pooling(kernel_size, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride = kernel, padding!=0
    pooling(kernel_size, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding=0
    pooling(kernel_size, stride= stride, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
    #stride != kernel, padding!=0
    pooling(kernel_size, stride= stride, padding= padding, dim = 3) # ceil model model, 
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, dim = 3) # floor model model, 
    pooling(kernel_size, stride= stride, padding=padding, count_include_pad =True, dim = 3) # ceil model model, none count_include_pad
    pooling(kernel_size, stride= stride, padding=padding, ceil_mode=True, count_include_pad =False, dim = 3) # ceil model model, none count_include_pad
