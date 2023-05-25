import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from model.cnn import CNN
from common.utils import accuracy
from common.meter import TrainMeters
from losses import loss_coteaching, setloss


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class coteaching:
    def __init__(
            self, 
            args, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):


        self.lr = args.lr
        self.args = args
        if args.forget_rate is None:
            forget_rate = (args.closeset_ratio + args.openset_ratio/(1-args.openset_ratio)) / ( 1+ args.openset_ratio/(1-args.openset_ratio))
        else:
            forget_rate = args.forget_rate 

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * args.epochs
        self.beta1_plan = [mom1] * args.epochs

        for i in range(args.epoch_decay_start, args.epochs):
            self.alpha_plan[i] = float(args.epochs - i) / (args.epochs - args.epoch_decay_start) * self.lr
            self.beta1_plan[i] = mom2

        self.device = device
        self.epochs = args.epochs

        # define drop rate schedule
        self.rate_schedule = np.ones(args.epochs) * forget_rate
        self.rate_schedule[:args.num_gradual ] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        # model
        self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model1.to(device)
        self.model2.to(device)
        

        self.optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.lr)
        self.loss_fn = loss_coteaching
        
        self.adjust_lr = args.adjust_lr
        self.TrainMeters = TrainMeters()
        

    def evaluate(self, test_loader):
        # print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        for images, labels, _, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        correct2 = 0
        total2 = 0
        for images, labels, _, _ in test_loader:
            images = Variable(images).to(self.device)
            logits2 = self.model2(images)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)
            total2 += labels.size(0)
            correct2 += (pred2.cpu() == labels).sum()

        acc1 = 100 * float(correct1) / float(total1)
        acc2 = 100 * float(correct2) / float(total2)
        return acc1, acc2

    def train(self, train_loader, epoch, trainset):
        # print('Training ...')
        self.model1.train()  
        self.model2.train()
        self.adjust_learning_rate(self.optimizer1, epoch)
        self.adjust_learning_rate(self.optimizer2, epoch)

        epoch_loss_fn = setloss(self.args, epoch)
        self.TrainMeters.reset()

        for (images, labels, _, truth) in train_loader:

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)
            truth = Variable(truth).to(self.device)
            
            logits1 = self.model1(images)
            logits2 = self.model2(images)

            self.TrainMeters.update(logits1, labels, truth)
            self.TrainMeters.update(logits2, labels, truth)
           
            loss_1, loss_2 = self.loss_fn(logits1, logits2, labels, \
                self.rate_schedule[epoch], epoch_loss_fn)
            self.optimizer1.zero_grad()
            loss_1.backward()
            self.optimizer1.step()
            self.optimizer2.zero_grad()
            loss_2.backward()
            self.optimizer2.step()

        return self.TrainMeters.return_val()
        
    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1