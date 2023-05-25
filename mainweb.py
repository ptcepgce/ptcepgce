# from distutils.command.config import config
import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
from common.core import accuracy, evaluate
from common.builder import *
from common.utils import *
from common.meter import AverageMeter, MaxMeter
from common.logger import Logger, print_to_logfile, print_to_console


from common.plotter import plot_results

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import math
from losses import *



def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)


class ResNet(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, pretrained=True, activation='tanh', classifier='linear'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.fc, init_method='He')
        # elif classifier.startswith('mlp'):
        #     sf = float(classifier.split('-')[1])
        #     fc = MLPHead(feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')


    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.fc(x)

        return logits
    

def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)
                

def build_logger(params):
    logger_root = f'Results/{params.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logger_root = f'Results/{params.dataset}/{params.net}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(logger_root, params.loss  +'_'+  f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir



    
def build_dataset_loader(params):
    assert not params.dataset.startswith('cifar')
    
    if params.dataset.startswith('web'):
        transform = build_transform(rescale_size=512, crop_size=448)
        dataset = build_webfg_dataset(os.path.join(params.data_root, params.dataset), transform['train'], transform['test'], params.number)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
    train_loader = DataLoader(dataset['train'], batch_size=params.bs, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader


def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, min(11, len(lines)+1)):
        line = lines[-idx].strip()
        epoch,  test_acc = line.split(' | ')[0],line.split(' | ')[-3]
        ep = int(epoch.split(': ')[1])
        valid_epoch.append(ep)
        # assert ep in valid_epoch, ep
        if '/' not in test_acc:
            test_acc_list.append(float(test_acc.split(': ')[1]))
        else:
            test_acc1, test_acc2 = map(lambda x: float(x), test_acc.split(': ')[1].lstrip('(').rstrip(')').split('/'))
            test_acc_list.append(test_acc1)
            test_acc_list2.append(test_acc2)
    if len(test_acc_list2) == 0:
        test_acc_list = np.array(test_acc_list)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f}, std: {test_acc_list.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}')
        return {'mean': test_acc_list.mean(), 'std': test_acc_list.std(), 'valid_epoch': valid_epoch}
    else:
        test_acc_list = np.array(test_acc_list)
        test_acc_list2 = np.array(test_acc_list2)
        print(valid_epoch)
        print(f'mean: {test_acc_list.mean():.2f} , std: {test_acc_list.std():.2f}')
        print(f'mean: {test_acc_list2.mean():.2f} , std: {test_acc_list2.std():.2f}')
        print(f' {test_acc_list.mean():.2f}±{test_acc_list.std():.2f}  ,  {test_acc_list2.mean():.2f}±{test_acc_list2.std():.2f} ')
        return {'mean1': test_acc_list.mean(), 'std1': test_acc_list.std(),
                'mean2': test_acc_list2.mean(), 'std2': test_acc_list2.std(),
                'valid_epoch': valid_epoch}
        
def wrapup_training(result_dir, best_accuracy):
    stats = get_baseline_stats(f'{result_dir}/log.txt')
    with open(f'{result_dir}/result_stats.txt', 'w') as f:
        f.write(f"valid epochs: {stats['valid_epoch']}\n")
        if 'mean' in stats.keys():
            f.write(f"mean: {stats['mean']:.4f}, std: {stats['std']:.4f}\n")
        else:
            f.write(f"mean1: {stats['mean1']:.4f}, std2: {stats['std1']:.4f}\n")
            f.write(f"mean2: {stats['mean2']:.4f}, std2: {stats['std2']:.4f}\n")
    os.rename(result_dir, f"{result_dir}-bestAcc_{best_accuracy:.4f}-lastAcc_{stats['mean']:.4f}")


def get_test_acc(acc):
        return (acc[0] + acc[1]) / 2. if isinstance(acc, tuple) else acc
       
def main(cfg, device):
    init_seeds(cfg.seed)
    assert cfg.dataset.startswith('web') or cfg.dataset.startswith('food')
    logger, result_dir = build_logger(cfg)
    # from model.resnetcut import ResNet18cut, load_pretrained,  load_pretrained_withfc
    # model = ResNet18cut(num_classes = cfg.n_classes).to(device) 
    # load_pretrained(model, cfg.net)

    model = ResNet(cfg.net, cfg.n_classes, pretrained=True ).to(device)  
    # model = torchvision.models.__dict__['resnet18'](pretrained=True)
    if len(cfg.gpu)>1:
        model =  torch.nn.DataParallel(model)
    
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9, nesterov=True)
    if cfg.schedule == 'max':
        lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, threshold=1e-6, verbose=True)
    elif cfg.schedule == 'cosine':
        lr_plan = [cfg.lr] * cfg.epochs
        for i in range(cfg.epochs):
            lr_plan[i] = 0.5 * cfg.lr * (1 + math.cos((i + 1) * math.pi / (cfg.epochs + 1)))  # cosine decay
    else:
        raise AssertionError(f'lr decay method: {cfg.schedule} must be max or cosine.')
    
    dataset, train_loader, test_loader = build_dataset_loader(cfg)
    logger.msg(f"Categories: {cfg.n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")


    best_accuracy, best_epoch = -1., 0

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        lossfn = setloss(cfg, epoch)
        start_time = time.time()
        test_total = 0
        test_correct = 0        
        model.train()
        if cfg.schedule == 'cosine':
            adjust_lr(optimizer, lr_plan[epoch])
        for i, (images, labels, _) in enumerate(train_loader):

            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            N = labels.size(0)
            # Forward + Backward + Optimize
            logits = model(images)
            
            loss = lossfn(logits, labels, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        for images, labels, _ in test_loader:
            images = Variable(images).to(device)
            logits = model(images)
            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (pred.cpu() == labels).sum()       
        test_accuracy = 100 * float(test_correct) / float(test_total)   
        
        if cfg.schedule =='max': 
            lr_schedule.step(test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        runtime = time.time() - start_time

        logger.info(f'epoch: {(epoch + 1):>3d} | '
                f'test accuracy: {test_accuracy:>6.3f} | '
                f'epoch runtime: {runtime:6.2f} sec | '
                f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt', layout='1x1')

    wrapup_training(result_dir, best_accuracy)


def setloss(params, epoch):
    # if params.dataset.startswith('web'):
    if params.loss == 'ce':
        loss = CE()
    elif params.loss == 'mae':
        loss = MAE(num_classes=params.n_classes)
    elif params.loss == 'sce':
        loss = SCELoss(dataset=params.dataset, num_classes=params.n_classes)
    elif params.loss == 'gce':
        loss = GCELoss(num_classes=params.n_classes)  
    elif params.loss == 'ncerce':
        loss = NCEandRCE( params.n_classes, 1, num_classes=params.n_classes)  #  params.n_classes/10, 0.1
    elif params.loss == 'tce':
        loss = TCE( n = 50)   # 50 
    elif params.loss == 'ptce':
        loss = PTCE( n = params.n_classes, t = params.t, epoch= epoch)            
    elif params.loss == 'pgce':
        loss = PGCE( t = params.t, epoch= epoch)  
    elif params.loss == 'ptceplus':
        loss = PTCEplus( n = params.n_classes, t = params.t, epoch= epoch) 
    elif params.loss == 'pgceplus':
        loss = PGCEplus( t = params.t, epoch= epoch) 
        
    return loss            
    # elif params.dataset.startswith('food'):

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='web-bird') 

    parser.add_argument('--gpu', type=str,  default='0')
    parser.add_argument('--net', type=str, default='resnet18')  # or resnet50
    parser.add_argument('--bs', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--schedule', type=str, default='cosine', help=' max or cosine')   #  lr-decay
    parser.add_argument('--weight-decay', type=float, default=1e-5)
   
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--loss', type=str, default='ce') 
    # parser.add_argument('--aug', type=str, default='w')   #  default weak  
    parser.add_argument('--data_root', type=str, default='~/dataroot') 
    
    parser.add_argument('--log', type=str, default='')
    
    parser.add_argument('--anno', type=str, default='', help='annotation') 
    parser.add_argument('--seed', type=int, default=1, help='seed to initial') 
    parser.add_argument('--number', type=int, default=None, help='the number of samples in each class, default None') 
    parser.add_argument('--t', type=int, default=5, help='epoch to 1') 

    args = parser.parse_args() 


    config = args
   
    if config.dataset.endswith('bird'):
        config.n_classes=200
    elif config.dataset.endswith('aircraft'):
        config.n_classes=100
    elif config.dataset.endswith('car'):    
        config.n_classes=196
    if config.net == 'resnet18':
        config.weight_decay=1e-5
        config.bs=64
    elif config.net == 'resnet50':
        config.weight_decay=5e-4
        config.bs=30
    print(config)
    return config

#   CUDA_VISIBLE_DEVICES=0  python mainweb.py  --bs 30  --net resnet50   --loss pgceplus    --t 5  --log cosine_t5  --dataset web-bird

if __name__ == '__main__':

    params = parse_args()
    dev = torch.device('cuda')
    print(dev)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
