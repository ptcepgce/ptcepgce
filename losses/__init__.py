from .loss_jocor import loss_jocor

from .loss_coteaching import loss_coteaching

from .loss_other import SCELoss, GCELoss, DMILoss, NCEandRCE, MAE, NLNL, TCE,  PTCE, PTCEplus, CE, PGCE, PGCEplus
from .loss_other import AExpLoss, AUELoss, AGCELoss


__all__ = ('loss_jocor',  'loss_coteaching', 
           'SCELoss', 'GCELoss', 'DMILoss', 'NCEandRCE', 'MAE', 'NLNL',  'TCE',  'PTCE', 'PTCEplus',  'CE', 'PGCE', 'PGCEplus',
            'AExpLoss', 'AUELoss', 'AGCELoss', 

           )

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
        if params.dataset == 'cifar100':
            loss = TCE( n = 6)      # cifar100 ————> 6 
        else:
            loss = TCE( n = 50)   
    # elif params.loss == 'tceplus':
    #     loss = TCEplus( n = 6)
    elif params.loss == 'ptce':
        loss = PTCE( n = params.n_classes, t = params.t, epoch= epoch)            
    elif params.loss == 'pgce':
        loss = PGCE( t = params.t, epoch= epoch)  
    elif params.loss == 'ptceplus':
        loss = PTCEplus( n = params.n_classes, t = params.t, epoch= epoch) 
    elif params.loss == 'pgceplus':
        loss = PGCEplus( t = params.t, epoch= epoch)  
        
    return loss    