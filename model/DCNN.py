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
    def __init__(self,k=3,lam=50):
        super(DLP_Loss, self).__init__()
        self.k = k
        self.lam = lam

    def forward(self,feture,scores,target):
        '''

        :param input: tensor shape(N,C)  FC layer scores
        :param target: tensor N
        :return:
        '''

        # softmax loss
        loss = func.cross_entropy(scores,target)
        #print(loss.item())
        N = feture.shape[0]
        # locality preserving loss
        for i in range(N):
            nums = self.kNN(i,feture,target)
            for j in range(len(nums)):
                loss += self.lam * 0.5 * func.mse_loss(feture[i],1 / self.k * feture[nums[j]],size_average=False)
        #print(loss.item())
        return loss

    def kNN(self,n,input,target):
        dict = {}
        tmp = input.shape[1]
        length = len(target)
        for i in range(length):
            if n != i and target[n] == target[i]:
                dist = func.pairwise_distance(input[n].view(tmp,-1),input[i].view(tmp,-1)).sum()
                dict[i] = dist
        dict = sorted(dict.items(),key=lambda item:item[1])
        nums = []
        for i in range(len(dict)):
            if i < self.k:
                nums.append(dict[i][0])
            else:
                return nums
        return nums

class DLP_CNN(nn.Module):
    def __init__(self,args):
        self.args = args
        super(DLP_CNN,self).__init__()
        self.conv1 =  nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(64,96,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(36864,2000)
        self.fc2 = nn.Linear(2000,args.output_classes)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv2(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv3(input)
        input = self.relu(input)
        input = self.conv4(input)
        input = self.relu(input)
        input = self.maxpool(input)
        input = self.conv5(input)
        input = self.relu(input)
        input = self.conv6(input)
        input = self.relu(input)
        input = self.flatten(input)
        input = self.fc1(input)
        input = self.relu(input)
        feature = input
        input = self.fc2(input)
        return input,feature



def get_DLP_CNN(args):
    model = DLP_CNN(args)
    return model


# the CPU type
dtype = torch.FloatTensor
# the GPU type
#dtype = torch.cuda.FloatTensor



if __name__ == '__main__':
    args = []
    model = DLP_CNN(args).type(dtype)
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
    for t in range(100):
        scores = model(images)
        myloss = loss(scores,targets)
        print(myloss.item()) # for pytorch0.5 change myloss.data[0] to myloss.item()
        optimizer.zero_grad()
        myloss.backward()
        optimizer.step()