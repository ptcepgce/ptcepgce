import torch
from easydict import EasyDict


class MaxMeter(object):
    def __init__(self):
        self.val = 0
    def reset(self):
        self.val = 0
    def update(self, val):
        self.val = max(self.val, val)
      
        
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count+1e-8)




class AvgcatMeter(object):
    def __init__(self):
        self.count = 0
        self.val = None
        self.avg = None
        self.sum = None  
        self.cat = None
        # cat: case_num * count, along dim=1,
        
    def update(self, val, n=1, dim=1):
        val = val.clone().detach().cpu().view(val.size(0),1)
        if self.val is None:
            self.val = val
            self.sum = self.val*n
            self.count = n
            self.avg =  self.sum / (self.count+1e-8)
            self.cat = self.avg
        else: 
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / (self.count+1e-8)
            self.cat = torch.cat( (self.cat, self.avg), dim=1 )
    
    def __len__(self):
        return self.cat.size(1)
    

    
    
    

class GeneralTrainMeters(object):
    def __init__(self):
        self.meters = EasyDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_accuracy'] = AverageMeter()

    def update(self, val_l, val_acc, n_l, n_acc=None):
        if n_acc is None:
            n_acc = n_l
        self.meters.train_loss.update(val=val_l, n=n_l)
        self.meters.train_accuracy.update(val=val_acc, n=n_acc)

    def reset(self, exclude=None):
        if exclude != 'train_loss':
            self.meters.train_loss.reset()
        if exclude != 'train_accuracy':
            self.meters.train_accuracy.reset()


class TrainMeters(object):
    def __init__(self):
        self.reset()
        
    def update(self, logits, labels, truth):
        _, pred = torch.max(logits.data, 1)
        self.total += labels.size(0)
        self.total_correct += (pred == labels).cpu().sum()
        self.clean_correct += (pred == labels)[labels==truth].cpu().sum()
        self.noise_correct += (pred == labels)[labels!=truth].cpu().sum()      
        
    def reset(self,):
        self.total = 0
        self.clean_correct = 0
        self.noise_correct = 0
        self.total_correct = 0
    
    def return_val(self,):
        return self.clean_correct, self.noise_correct, self.total_correct, self.total