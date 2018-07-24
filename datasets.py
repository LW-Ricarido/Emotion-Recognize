import os
import cv2
import torch
import numpy
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data
import argparse

def get_train_loader(args):
    dataset = RAFTrainSet(args)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True
    )

def get_test_loader(args):
    dataset = RAFTestSet(args)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True
    )

class RAFTrainSet(data.Dataset):
    def __init__(self,args):
        self.images = list()
        self.targets = list()
        self.args = args

        #
        #
        #
        lines = open(args.train_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(args.data_dir,path))
            self.targets.append(int(label) - 1)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image,target


    def __len__(self):
        return len(self.targets)

class RAFTestSet(data.Dataset):
    def __init__(self,args):
        self.images = list()
        self.targets = list()
        self.args = args

        #
        #
        #
        lines = open(args.test_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(os.path.join(args.data_dir, path))
            self.targets.append(int(label) - 1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image,target,self.images[index]

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mytest")
    parser.add_argument('--data_dir',type=str,default='DataSet/RAF/basic/Image/aligned/')
    parser.add_argument('--size',type=int,default=100)
    parser.add_argument('--train_list',type=str,default='DataSet/RAF/basic/train_set')
    args = parser.parse_args()
    mytest = RAFTrainSet(args)
    print(mytest.__len__())
    image,target = mytest.__getitem__(2)
    print(image.shape)
    image = transforms.ToPILImage()(image).convert('RGB')

    image.show()
