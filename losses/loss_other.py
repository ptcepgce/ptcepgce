# Thanks to https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py

import torch
import torch.nn.functional as F
import numpy as np


class MAE(torch.nn.Module):  # \ell_1 = 1-p
    def __init__(self, num_classes, scale=1.0):
        super(MAE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels, reduction = 'mean'):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        if reduction == 'mean': return self.scale *mae.mean()
        elif reduction == 'none': return self.scale * mae
        else: raise ValueError('reduction should be mean or none')
        

class CE(torch.nn.Module):
    def __init__(self, ):
        super(CE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def forward(self, logits, labels, reduction='mean'):
        return F.cross_entropy(logits, labels,reduction=reduction)
    
class TCE(torch.nn.Module):
    def __init__(self, n):
        super(TCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n = n

    def forward(self, logits, labels, reduction='mean'):
        if self.n == 0 :
            return torch.nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
        prob = logits.softmax(-1)
        prob_y = prob.gather(1,labels.view(-1,1)).view(-1)
        x = (1-prob_y)
        loss = torch.zeros(labels.size()).to(self.device)
        pt = 1
        for i in range(1,self.n+1):
            pt = pt*x
            loss += pt/i
        if reduction == 'mean': return loss.mean()
        elif reduction == 'none': return loss
        else: raise ValueError('reduction should be mean or none')

class PTCE(torch.nn.Module):
    def __init__(self, n, t, epoch):
        super(PTCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n = n
        self.t = t
        if epoch<1: self.lossfn = TCE( 0 )
        elif epoch>=1 and epoch<=self.t: 
            self.lossfn  = TCE( int(self.n *(self.t-epoch)/self.t + 1) )
        elif epoch>self.t:
            self.lossfn  = TCE( 1 )
            
    def forward(self, logits, labels, reduction='mean'):
        return self.lossfn(logits, labels, reduction)
    
    
class GCE(torch.nn.Module):
    def __init__(self,  q=0.7):
        super(GCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.q = q
        
    def forward(self, logits, labels, reduction='mean'):
        if self.q ==0: 
            return torch.nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
        pred = F.softmax(logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        p = pred.gather( 1, labels.view(labels.size(0),1)).view(-1)
        gce = (1. - torch.pow(p, self.q)) / self.q
        
        if reduction == 'mean': return gce.mean()
        elif reduction == 'none': return gce
        else: raise ValueError('reduction should be mean or none') 
                
class PGCE(torch.nn.Module):
    def __init__(self, t, epoch):
        super(PGCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.t = t
        if epoch<1: self.lossfn = GCE( 0 )
        elif epoch>=1 and epoch<=self.t: 
            self.lossfn  = GCE( epoch/self.t )
        elif epoch>self.t:  
            self.lossfn  = GCE( 1 ) 
        # print(torch.cuda.is_available())

    def forward(self, logits, labels, reduction='mean'):
        return self.lossfn(logits, labels, reduction)

class GCEplus(torch.nn.Module):
    def __init__(self,q):
        super(GCEplus, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.q = q

    def forward(self, logits, labels, reduction='mean'):
        if self.q == 0 :
            return torch.nn.CrossEntropyLoss(reduction=reduction)(logits, labels) + MAE(logits.size(-1))(logits, labels,reduction)
        elif self.q == 1 :
            return MAE(logits.size(-1))(logits, labels, reduction)
        else:
            pred = F.softmax(logits, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            p = pred.gather( 1, labels.view(labels.size(0),1)).view(-1)
            loss = (1. - torch.pow(p, self.q)) / self.q
            loss += MAE(logits.size(-1))(logits, labels, 'none')
            if reduction == 'mean': return loss.mean()
            elif reduction == 'none': return loss
            else: raise ValueError('reduction should be mean or none')
                    
class PGCEplus(torch.nn.Module):
    def __init__(self,  t, epoch):
        super(PGCEplus, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.t = t            
        if epoch<1: self.lossfn = GCEplus( 0 )
        elif epoch>=1 and epoch<=self.t: 
            self.lossfn  = GCEplus( epoch/self.t )
        elif epoch>self.t:  
            self.lossfn  = GCEplus( 1 ) 
            
    def forward(self, logits, labels, reduction='mean'):
        return self.lossfn(logits, labels, reduction)
    
    
    
class TCEplus(torch.nn.Module):
    def __init__(self, n):
        super(TCEplus, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n = n

    def forward(self, logits, labels, reduction='mean'):
        if self.n == 0 :
            return torch.nn.CrossEntropyLoss(reduction=reduction)(logits, labels) + MAE(logits.size(-1))(logits, labels,reduction)
        elif self.n == 1 :
            return MAE(logits.size(-1))(logits, labels, reduction)
        else:
            prob = logits.softmax(-1)
            prob_y = prob.gather(1,labels.view(-1,1)).view(-1)
            x = (1-prob_y)
            loss = torch.zeros(labels.size()).to(self.device)
            pt = 1
            for i in range(1,self.n+1):
                pt = pt*x
                loss += pt/i
            loss += MAE(logits.size(-1))(logits, labels, 'none')
            if reduction == 'mean': return loss.mean()
            elif reduction == 'none': return loss
            else: raise ValueError('reduction should be mean or none')
            
            
class PTCEplus(torch.nn.Module):
    def __init__(self, n, t, epoch):
        super(PTCEplus, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n = n
        self.t = t
        if epoch<1: self.lossfn = TCEplus( 0 ) #TCE( 1 )
        elif epoch>=1 and epoch<=self.t: 
            self.lossfn  = TCEplus( int(self.n *(self.t-epoch)/self.t + 1) )
        elif epoch>self.t:
            self.lossfn  = TCEplus( 1 )
            
    def forward(self, logits, labels, reduction='mean'):
        return self.lossfn(logits, labels, reduction)



#---------------------------------------------------------------------------------------------------------

        
class SCELoss(torch.nn.Module):
    def __init__(self, dataset, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if dataset == 'cifar10':
            self.alpha, self.beta = 0.1, 1.0
        elif dataset == 'cifar100':   #  6.0, 0.1
            self.alpha, self.beta = 0.1, 1  
        elif dataset.startswith('web'):    #  6.0, 0.1
            self.alpha, self.beta = 0.1, 1  
        elif dataset.startswith('food'):    #  6.0, 0.1
            self.alpha, self.beta = 0.1, 1    
        elif dataset.startswith('miniwebvision'):    #  6.0, 0.1
            self.alpha, self.beta = 0.1, 1 
        self.num_classes = num_classes
        

    def forward(self, pred, labels, reduction = 'mean'):
        # CCE
        ce = torch.nn.CrossEntropyLoss(reduction='none')(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce
        
        if reduction == 'mean': return loss.mean()
        elif reduction == 'none': return loss
        else: raise ValueError('reduction should be mean or none')



class GCELoss(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GCELoss, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.q = q
        # print(torch.cuda.is_available())

    def forward(self, pred, labels, reduction='mean'):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        
        if reduction == 'mean': return gce.mean()
        elif reduction == 'none': return gce
        else: raise ValueError('reduction should be mean or none')


class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)
    
    
class NCE(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels, reduction='mean'):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))

        if reduction == 'mean': return self.scale * nce.mean()
        elif reduction == 'none': return self.scale * nce
        else: raise ValueError('reduction should be mean or none')
   
   
class RCE(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(RCE, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels, reduction='mean'):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        if reduction == 'mean': return self.scale * rce.mean()
        elif reduction == 'none': return self.scale * rce
        else: raise ValueError('reduction should be mean or none')
    
class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NCE(scale=alpha, num_classes=num_classes)
        self.rce = RCE(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels, reduction='mean'):
        return self.nce(pred, labels, reduction) + self.rce(pred, labels, reduction)

    

class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(self.device).random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss
    









class AGCELoss(torch.nn.Module):
    def __init__(self, num_classes=100, a=0.6, q=0.6, scale=1.):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean() * self.scale

class AUELoss(torch.nn.Module):
    def __init__(self, num_classes=100, a=5.5, q=3, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean() * self.scale

# class ANormLoss(torch.nn.Module):
#     def __init__(self, num_classes=10, a=1.5, p=0.9, scale=1.0):
#         super(ANormLoss, self).__init__()
#         self.num_classes = num_classes
#         self.a = a
#         self.p = p
#         self.scale = scale

#     def forward(self, pred, labels):
#         pred = F.softmax(pred, dim=1)
#         pred = torch.clamp(pred, min=1e-5, max=1)
#         label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
#         loss = torch.sum(torch.pow(torch.abs(self.a * label_one_hot-pred), self.p), dim=1) - (self.a-1)**self.p
#         return loss.mean() * self.scale / self.p


class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=100, a=2.5, scale=1.0):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean() * self.scale
    

