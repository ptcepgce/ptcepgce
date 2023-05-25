import os
import sys
import pathlib
import time
import datetime
import argparse
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import warnings
warnings.filterwarnings('ignore')
from common.core import accuracy, evaluate
from common.builder import *
from common.utils import *
from common.meter import AverageMeter
from common.logger import Logger, print_to_logfile, print_to_console

from common.plotter import plot_results

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np



class TwoDataTransform(object):
    def __init__(self, transform_weak, transform_strong):
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak(sample)
        x_s = self.transform_strong(sample)
        return x_w1,  x_s

class ThreeDataTransform(object):
    def __init__(self, transform_weak1, transform_weak2, transform_strong):
        self.transform_weak1 = transform_weak1
        self.transform_weak2 = transform_weak2
        self.transform_strong = transform_strong

    def __call__(self, sample):
        x_w1 = self.transform_weak1(sample)
        x_w2 = self.transform_weak2(sample)
        x_s = self.transform_strong(sample)
        return x_w1, x_w2, x_s
    
    


def save_current_script(log_dir):
    current_script_path = __file__
    shutil.copy(current_script_path, log_dir)



def build_logger(params):
    if params.ablation:
        logger_root = f'Ablation/{params.synthetic_data}'
    else:
        logger_root = f'Results/{params.synthetic_data}'
    logger_root = str(params.net) + logger_root
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    percentile = int(params.closeset_ratio * 100)
    noise_condition = f'symm_{percentile:2d}' if params.noise_type == 'symmetric' else f'asym_{percentile:2d}'
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if params.ablation:
        result_dir = os.path.join(logger_root, noise_condition, params.method + '_' + params.project, f'{params.log}-{logtime}')
    else:
        result_dir = os.path.join(logger_root, noise_condition, params.method, f'{params.log}-{logtime}')
    logger = Logger(logging_dir=result_dir, DEBUG=True)
    logger.set_logfile(logfile_name='log.txt')
    # save_config(params, f'{result_dir}/params.cfg') 
    save_params(params, f'{result_dir}/params.json', json_format=True)
    save_current_script(result_dir)
    logger.msg(f'Result Path: {result_dir}')
    return logger, result_dir




    
def build_dataset_loader(params):
    assert params.dataset.startswith('cifar')
    transform = build_transform(rescale_size=params.rescale_size, crop_size=params.crop_size)
    cifar_tensor = torchvision.transforms.ToTensor()
    cifar_normalize = transform['cifar_test']
    cifar_weak = transform['cifar_train']
    cifar_strong = transform['cifar_train_strong_aug']
    cifar_train_trans = {'t':cifar_tensor, 
                   'n':cifar_normalize,
                   'w':cifar_weak,
                   's':cifar_strong}
    cifar_test_trans = {'t':cifar_tensor, 
                   'n':cifar_normalize,
                   'w':cifar_normalize,
                   's':cifar_normalize}
    datadir = os.path.join(params.data_root, params.dataset)
    
    
    if params.dataset == 'cifar100':
        if len(params.aug)==1:
            dataset = build_cifar100n_dataset(datadir, cifar_train_trans[params.aug], cifar_test_trans[params.aug], noise_type=params.noise_type, 
                                              openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif len(params.aug)==2:
            dataset = build_cifar100n_dataset(datadir, TwoDataTransform(cifar_train_trans[params.aug[0]], cifar_train_trans[params.aug[1]]), cifar_test_trans[params.aug[-1]], 
                                              noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif len(params.aug)==3:
            dataset = build_cifar100n_dataset(datadir, ThreeDataTransform(cifar_train_trans[params.aug[0]], cifar_train_trans[params.aug[1]], cifar_train_trans[params.aug[2]]), 
                                              cifar_test_trans[params.aug[-1]], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
    elif params.dataset == 'cifar10':
        if len(params.aug)==1:
            dataset = build_cifar10n_dataset(datadir, cifar_train_trans[params.aug], cifar_test_trans[params.aug], noise_type=params.noise_type, 
                                             openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif len(params.aug)==2:
            dataset = build_cifar10n_dataset(datadir, TwoDataTransform(cifar_train_trans[params.aug[0]], cifar_train_trans[params.aug[1]]), cifar_test_trans[params.aug[-1]], 
                                             noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
        elif len(params.aug)==3:
            dataset = build_cifar10n_dataset(datadir, ThreeDataTransform(cifar_train_trans[params.aug[0]], cifar_train_trans[params.aug[1]], cifar_train_trans[params.aug[2]]), 
                                              cifar_test_trans[params.aug[-1]], noise_type=params.noise_type, openset_ratio= params.openset_ratio, closeset_ratio=params.closeset_ratio)
    else:
        raise AssertionError(f'{params.dataset} dataset is not supported yet.')
        
    
    train_loader = DataLoader(dataset['train'], batch_size=params.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset['test'], batch_size=16, shuffle=False, num_workers=8, pin_memory=False)
    return dataset, train_loader, test_loader



def get_baseline_stats(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    test_acc_list = []
    test_acc_list2 = []
    valid_epoch = []
    # valid_epoch = [191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
    for idx in range(1, min(11,len(lines)+1)):
        line = lines[-idx].strip()
        epoch, test_acc = line.split(' | ')[0], line.split(' | ')[-3]
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
    assert cfg.dataset.startswith('cifar')
    logger, result_dir = build_logger(cfg)

    import algorithm
    n_classes = int(cfg.n_classes * (1 - cfg.openset_ratio))
    
    dataset, train_loader, test_loader = build_dataset_loader(cfg)
    

    model = algorithm.__dict__[cfg.method](cfg, input_channel=3, num_classes= n_classes)   
    
    logger.msg(f"Categories: {n_classes}, Training Samples: {dataset['n_train_samples']}, Testing Samples: {dataset['n_test_samples']}")
    logger.msg(f"Noise Type: {dataset['train'].noise_type}, Openset Noise Ratio: {dataset['train'].openset_noise_ratio}, Closedset Noise Ratio: {dataset['train'].closeset_noise_rate}")
    logger.msg(f'Optimizer: {cfg.opt}')


    best_accuracy, best_epoch = -1.0, None

    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(0, cfg.epochs):
        start_time = time.time()


        clean_correct, noise_correct, total_correct, total = model.train(train_loader, epoch, dataset['train'] )

        test_accuracy = get_test_acc(model.evaluate(test_loader))
        

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1

        runtime = time.time() - start_time

        logger.info(f'epoch: {epoch + 1:>3d} | '
                f'train clean correct: {clean_correct:>6.1f} | '
                f'train noise correct: {noise_correct:>6.1f} | '
                f'train total correct: {total_correct:>6.1f} | '
                f'train total: {total:>6.1f} | '
                f'test accuracy: {test_accuracy:>6.3f} | '
                f'epoch runtime: {runtime:6.2f} sec | '
                f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}')
        plot_results(result_file=f'{result_dir}/log.txt', layout='1x2')


    wrapup_training(result_dir, best_accuracy)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-data', type=str, default='cifar100nc')  # 'cifar100nc'   cifar80no  
    parser.add_argument('--dataset', type=str, default='cifar100') 
    parser.add_argument('--rescale_size', type=int, default=32) 
    parser.add_argument('--crop_size', type=int, default=32) 
    parser.add_argument('--input_channel', type=int, default=3) 
    parser.add_argument('--aug', type=str, default='w')     
    
    parser.add_argument('--noise-type', type=str, default='symmetric')   #  symmetric  asymmetric
    parser.add_argument('--closeset_ratio', type=float, default='0.8')   #   closeset-ratio
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--net', type=str, default='cnn')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=str, default='linear')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--opt', type=str, default='adam')

    parser.add_argument('--epochs', type=int, default=200)
    
    parser.add_argument('--method', type=str, default='standard', help='standard, coteaching, jocor, ce, ptce, ...')
    parser.add_argument('--loss', type=str, default='ce')  # ptce, ...
    
    parser.add_argument('--data_root', type=str, default='~/dataroot')    
    
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--log', type=str, default='')

  
    parser.add_argument('--forget_rate', type=float,  default=None)
    parser.add_argument('--adjust_lr', type=int,  default=1)
    parser.add_argument('--epoch_decay_start', type=int,  default=80)
    
    parser.add_argument('--num_gradual', type=int,  default=10)
    parser.add_argument('--exponent', type=float,  default=1)
    
    parser.add_argument('--ablation', action='store_true')

    parser.add_argument('--seed', type=int, default=1, help='seed to initial') 
    parser.add_argument('--alpha', type=float, default=10.0, help='weight for the nce loss')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for the rce loss')  
    parser.add_argument('--n', type=int, default=100) 
    parser.add_argument('--t', type=int, default=20, help='epoch to 1')  
    
    
    args = parser.parse_args()


    config = args

    assert config.synthetic_data in ['cifar100nc', 'cifar80no']
    assert config.noise_type in ['symmetric', 'asymmetric']
    config.openset_ratio = 0.2 if config.synthetic_data == 'cifar80no' else 0
    if config.dataset == 'cifar100': config.n_classes=100 


    print(config)
    return config


if __name__ == '__main__':

    params = parse_args()
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(dev)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
    
