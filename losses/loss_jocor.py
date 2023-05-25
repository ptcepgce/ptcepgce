import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



eps = 1e-12

def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_jocor(y1, y2, labels, forget_rate, epoch_loss_fn, co_lambda=0.1):
    # loss_pick_1 = F.cross_entropy(y1, labels, reduce = False) * (1 - co_lambda)
    # loss_pick_2 = F.cross_entropy(y2, labels, reduce = False) * (1 - co_lambda)
    loss_pick_1 = epoch_loss_fn(y1, labels, reduction='none') * (1 - co_lambda)
    loss_pick_2 = epoch_loss_fn(y2, labels, reduction='none') * (1 - co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y1, y2,reduce=False) + co_lambda * kl_loss_compute(y2, y1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss = torch.mean(loss_pick[ind_update])

    return loss, loss
