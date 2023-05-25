# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import CNN
import numpy as np
from common.utils import accuracy
from common.meter import TrainMeters
from losses import loss_jocor,  setloss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class jocor:
    def __init__(self, args, input_channel, num_classes):

        # Hyper Parameters
        self.args = args
        self.batch_size = 128
        learning_rate = args.lr

        
        if args.forget_rate is None:
            forget_rate = (args.closeset_ratio + args.openset_ratio/(1-args.openset_ratio)) / ( 1+ args.openset_ratio/(1-args.openset_ratio))
        else:
            forget_rate = args.forget_rate
        
        
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.epochs
        self.beta1_plan = [mom1] * args.epochs

        for i in range(args.epoch_decay_start, args.epochs):
            self.alpha_plan[i] = float(args.epochs - i) / (args.epochs - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.epochs) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
 
        if args.dataset =='cifar100':  self.co_lambda = 0.85
        elif args.dataset =='cifar10':
            if args.closeset_ratio == 0.8 :
                self.co_lambda = 0.65
            elif args.closeset_ratio < 0.6 :
                self.co_lambda = 0.9
        print( 'co_lambda: ', self.co_lambda )
        self.epochs = args.epochs
        # self.train_dataset = train_dataset


        self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)


        self.model1.to(device)
        self.model2.to(device)

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()),
                                          lr=learning_rate)

        self.TrainMeters = TrainMeters()
        self.adjust_lr = args.adjust_lr
        self.loss_fn = loss_jocor

    # Evaluate the Model
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

    # Train the Model
    def train(self, train_loader, epoch, trainset):
        # print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode

        self.adjust_learning_rate(self.optimizer, epoch)

        self.TrainMeters.reset()
        epoch_loss_fn = setloss(self.args, epoch)
        for i, (images, labels, _, truth) in enumerate(train_loader):

            images = Variable(images).to(self.device)
            labels = Variable(labels).to(self.device)
            truth = Variable(truth).to(self.device)
            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            logits2 = self.model2(images)


            self.TrainMeters.update(logits1, labels, truth)
            self.TrainMeters.update(logits2, labels, truth)

            loss_1, loss_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], \
                epoch_loss_fn, self.co_lambda)
            

            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

           
        return self.TrainMeters.return_val()

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
