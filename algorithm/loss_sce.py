import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from model.cnn import CNN
from common.utils import accuracy
from losses import SCELoss
from common.meter import TrainMeters


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class sce:
    def __init__(
            self, 
            args, 
            input_channel: int = 3, 
            num_classes: int = 10,
        ):

        self.lr = args.lr
        self.args = args
        self.num_classes = num_classes
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

        # scratch
        self.model_scratch = CNN(input_channel=input_channel, n_outputs=num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.adjust_lr = args.adjust_lr

        # loss function
        self.criterion = SCELoss(dataset=args.dataset, num_classes=num_classes)
        self.TrainMeters = TrainMeters()
        
        
        
    def evaluate(self, test_loader):
        # print('Evaluating ...')
        self.model_scratch.eval()  # Change model to 'eval' mode

        correct = 0
        total = 0
        for images, labels,_ , _ in test_loader:
            images = Variable(images).to(self.device)
            logits = self.model_scratch(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()

        acc = 100 * float(correct) / float(total)
        return acc

    def train(self, train_loader, epoch, trainset):
        # print('Training ...')
        self.model_scratch.train()

        self.adjust_learning_rate(self.optimizer, epoch)
        
        self.TrainMeters.reset()
        for (images, labels, _, truth) in train_loader:
            images = Variable(images).to(self.device, non_blocking=True)
            labels = Variable(labels).to(self.device)
            truth = Variable(truth).to(self.device)
            
            logits = self.model_scratch(images)
            self.TrainMeters.update(logits, labels, truth)
            
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.TrainMeters.return_val()
    

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1
