import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import cv2
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        N, C, H, W = input.size()
        return input.view(N,-1)

class DLP_Loss(nn.Module):
    def __init__(self,k=1,lam=0.5,theta=2):
        self.k = k
        self.lam = lam
        self.theta = theta
        return

    def forward(self,input,target):
        '''

        :param input: tensor shape(N,C)  FC layer scores
        :param target: tensor N
        :return:
        '''
        N = input.shape[0]
        tmp = torch.Tensor.sum(torch.Tensor.exp(input),dim=1)
        loss = torch.Tensor.sum(torch.Tensor.log(tmp))
        loss -= torch.Tensor.sum(input[range(0,N),target])
        loss /= N
        return loss



# the CPU type
dtype = torch.FloatTensor
# the GPU type
#dtype = torch.cuda.FloatTensor

model = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=3,padding=1), # 64 x 100 x 100
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2), # 64 x 50 x 50
    nn.Conv2d(64,96,kernel_size=3,padding=1), # 96 x 50 x 50
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2), #96 x 25 x 25
    nn.Conv2d(96,128,kernel_size=3,padding=1), # 128 x 25 x 25
    nn.ReLU(inplace=True),
    nn.Conv2d(128,128,kernel_size=3,padding=1), # 128 x 25 x 25
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2,stride=2), # 128 x 11 x 11
    nn.Conv2d(128,256,kernel_size=3,padding=1), # 256 x 11 x 11
    nn.ReLU(inplace=True),
    nn.Conv2d(256,256,kernel_size=3,padding=1), # 256 x 11 x 11
    nn.ReLU(inplace=True),
    Flatten(),
    nn.Linear(30976,2000),
    nn.ReLU(inplace=True),
    nn.Linear(2000,7)
)


if __name__ == '__main__':
    model = model.type(dtype)
    loss = DLP_Loss().type(dtype)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    mytest = torch.randn(10,3,100,100)
    myy = torch.randn(10)
    for t in range(10):
        scores = model(mytest)
        myloss = loss(scores,myy)
        print(loss)
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()