import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

class Trainer:
    def __init__(self,args,model,criterion,logger):
        self.args = args
        self.decay = 1
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(
            model.parameters(),
            args.learn_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        self.nGPU = args.nGPU
        self.learning_rate = args.learn_rate

    def train(self,epoch,train_loader):
        n_batches = len(train_loader)

        acc_avg = 0

        loss_avg = 0
        total = 0

        model = self.model
        model.train()
        self.learning_rate(epoch)

        for i ,(input_tensor, target) in enumerate(train_loader):
            if self.args.mixup:
                input_tensor, target = shuff