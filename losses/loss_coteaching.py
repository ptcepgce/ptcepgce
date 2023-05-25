import torch 
import torch.nn.functional as F
import numpy as np

def loss_coteaching(y_1, y_2, t, forget_rate, epoch_loss_fn):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = epoch_loss_fn(y_1[ind_2_update], t[ind_2_update], reduction='none')
    loss_2_update = epoch_loss_fn(y_2[ind_1_update], t[ind_1_update], reduction='none')


    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

