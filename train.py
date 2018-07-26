import torch
import torch.optim as optim
import torch.nn as nn
import os
from torch.autograd import Variable
from utils.mixup import shuffle_minibatch
from sklearn.svm import LinearSVC

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
        self.svm = LinearSVC(random_state=0)
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate

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
                input_tensor, target = shuffle_minibatch(input_tensor,target,self.args,mixup=True)

            if self.nGPU > 0 :
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            target_var = Variable(target)
            if self.args.model == 'DLP_CNN':
                output,feature = model(input_var)
            else:
                output = model(input_var)

            if self.args.mixup:
                m = nn.LogSoftmax(dim=1)
                loss = -m(output) * target
                loss = torch.sum(loss) / 128
                _, target = torch.max(target.item(),1)
            if self.args.model == 'DLP_CNN':
                loss = self.criterion(feature,output,target_var)
            else:
                loss = self.criterion(output,target_var)

            acc ,_= self.accuracy(output.data,target,(1,3))
            acc_avg += acc * batch_size

            loss_avg += loss.item() * batch_size
            total += batch_size
            if i % 10 == 0:
                print("| Epoch[%d] [%d/%d]  Loss %1.4f  Acc %6.3f   LR %1.8f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    loss_avg / total,
                    acc_avg / total,
                    self.decay * self.learn_rate))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_avg /= total
        acc_avg /= total
        print("\n=> Epoch[%d]  Loss: %1.4f  Acc %6.3f  \n" % (
            epoch,
            loss_avg,
            acc_avg))
        torch.cuda.empty_cache()
        summary = dict()

        summary['acc'] = acc_avg
        summary['loss'] = loss_avg

        return summary


    def test(self, epoch, test_loader):

        n_batches = len(test_loader)

        acc_avg = 0

        total = 0

        model = self.model
        model.eval()
        out_f = open('results.txt', 'w')
        for i, (input_tensor, target, filenames) in enumerate(test_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            if self.args.model == 'DLP_CNN':
                output,feature = model(input_var)
            else:
                output = model(input_var)

            if self.args.save_result:
                _, predictions = output.topk(3, dim=1, )
                for filename, prediction in zip(filenames, predictions):
                    out_f.write(str(os.path.basename(filename)).split('.')[0] + ',' + ','.join(
                        [str(int(x)) for x in prediction]) + '\n')

            acc,_ = self.accuracy(output.data, target,(1,3))

            acc_avg += acc * batch_size

            total += batch_size
            if i % 1000 == 0:
                print("| Test[%d] [%d/%d]   Acc %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    acc_avg / total))

        acc_avg /= total

        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = acc_avg
        return summary

    def svm_classifier(self,epoch,train_loader,test_loader):
        n_batches = len(test_loader)

        acc_avg = 0

        total = 0

        model = self.model
        model.eval()
        out_f = open('results.txt', 'w')
        for i,(input_tensor,target) in enumerate(train_loader):
            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            input_var = Variable(input_tensor)
            if self.args.model == 'DLP_CNN':
                output,feature = model(input_var)
                self.svm.fit(feature,target)
            else:
                output = model(input_var)
                self.svm.fit(output,target)


        for i, (input_tensor, target, filenames) in enumerate(test_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            if self.args.model == 'DLP_CNN':
                output, feature = model(input_var)
                output = self.svm.predict(feature)
            else:
                output = model(input_var)

            # if self.args.save_result:
            #     _, predictions = output.topk(3, dim=1, )
            #     for filename, prediction in zip(filenames, predictions):
            #         out_f.write(str(os.path.basename(filename)).split('.')[0] + ',' + ','.join(
            #             [str(int(x)) for x in prediction]) + '\n')
            #
            # acc, _ = self.accuracy(output.data, target, (1, 3))
            if int(output[0]) == int(target[0]):
                acc = 100
            else:
                acc = 0
            acc_avg += acc * batch_size

            total += batch_size
            if i % 1000 == 0:
                print("| Test[%d] [%d/%d]   Acc %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    acc_avg / total))

        acc_avg /= total

        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = acc_avg
        return summary

    def accuracy(self,output,target,topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correcct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correcct_k = correcct[:k].view(-1).float().sum(0,keepdim=True)

            res.append(correcct_k.mul_(100.0 / batch_size)[0])
        return res

    def learning_rate(self,epoch):
        self.decay = 0.1 **((epoch - 1) // self.args.decay)
        learn_rate = self.learn_rate * self.decay
        if learn_rate < 1e-7:
            learn_rate = 1e-7
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate
