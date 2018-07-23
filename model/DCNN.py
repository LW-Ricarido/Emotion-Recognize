import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import argparse
import torchvision
from torch.autograd import Variable
from datasets import RAFTrainSet
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import cv2
import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        N, C, H, W = input.size()
        return input.view(N,-1)

class DLP_Loss(nn.Module):
    def __init__(self,k=1,lam=0.5):
        super(DLP_Loss, self).__init__()
        self.k = k
        self.lam = lam

    def forward(self,input,scores,target):
        '''

        :param input: tensor shape(N,C)  FC layer scores
        :param target: tensor N
        :return:
        '''
        N = input.shape[0]
        # softmax loss
        loss = func.cross_entropy(scores,target)

        # locality preserving loss
        for i in range(N):
            nums = self.kNN(i,input,target)
            for j in range(len(nums)):
                loss += 0.5 * 1 / self.k * func.mse_loss(input[i],input[nums[j]],size_average=True)
        return loss

    def kNN(self,n,input,target):
        dict = {}
        length = len(target)
        for i in range(length):
            if n != i and target[n] == target[i]:
                dist = func.pairwise_distance(input[n],input[i]).sum()
                dict[i] = dist
        dict = sorted(dict.items(),key=lambda item:item[1])
        nums = []
        for i in range(len(dict)):
            if i < self.k:
                nums.append(dict[i][0])
            else:
                return nums
        return nums

def DLP_CNN():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 64 x 100 x 100
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 64 x 50 x 50
        nn.Conv2d(64, 96, kernel_size=3, padding=1),  # 96 x 50 x 50
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 96 x 25 x 25
        nn.Conv2d(96, 128, kernel_size=3, padding=1),  # 128 x 25 x 25
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128 x 25 x 25
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 128 x 12 x 12  ?
        nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 x 12 x 12
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256 x 12 x 12
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(36864, 2000),
        nn.ReLU(inplace=True),
        nn.Linear(2000, 7)
    )
    return model


# the CPU type
dtype = torch.FloatTensor
# the GPU type
#dtype = torch.cuda.FloatTensor



if __name__ == '__main__':
    model = DLP_CNN().type(dtype)
    loss = DLP_Loss(k=3).type(dtype)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    parser = argparse.ArgumentParser(description="mytest")
    parser.add_argument('--data_dir', type=str, default='../DataSet/RAF/basic/Image/aligned/')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--train_list', type=str, default='../DataSet/RAF/basic/train_set')
    args = parser.parse_args()
    mytest = RAFTrainSet(args)
    index = torch.randint(0,1000,(10,))
    images = Variable(torch.randn(10,3,100,100))
    targets = Variable(torch.randn(10,)).long()
    for i in range(10):
        image,target = mytest.__getitem__(int(index[i]))
        images[i] = image
        targets[i] = target
    for t in range(10):
        scores = model(images)
        myloss = loss(images,scores,targets)
        print(myloss.item()) # for pytorch0.5 change myloss.data[0] to myloss.item()
        optimizer.zero_grad()
        myloss.backward()
        optimizer.step()