import torch

from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU
from configs import cfg_train

def get_optimizer(net):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg_train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg_train.learning_rate, weight_decay=cfg_train.weight_decay)
    elif cfg_train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg_train.learning_rate, momentum=cfg_train.momentum,
                                    weight_decay=cfg_train.weight_decay)
    else:
        raise NotImplementedError
    
    return optimizer

def get_scheduler(optimizer, iters_per_epoch):
    if cfg_train.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg_train.steps, cfg_train.gamma, iters_per_epoch, cfg_train.warmup, iters_per_epoch if cfg_train.warmup_iters is None else cfg_train.warmup_iters)
    elif cfg_train.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg_train.epoch * iters_per_epoch, eta_min = 0, warmup = cfg_train.warmup, warmup_iters = cfg_train.warmup_iters)
    else:
        raise NotImplementedError
    
    return scheduler

def get_loss_dict():
    loss_dict = {}

    loss_dict['cls_loss'] = {
        'op': SoftmaxFocalLoss(2),
        'weight': 1.0,
        'data_src': ('cls_out', 'cls_label')
    }

    loss_dict['relation_loss'] = {
        'op': ParsingRelationLoss(),
        'weight': cfg_train.sim_loss_w,
        'data_src': ('cls_out',)
    }

    loss_dict['relation_dis'] = {
        'op': ParsingRelationDis(),
        'weight': cfg_train.shp_loss_w,
        'data_src': ('cls_out',)
    }

    if cfg_train.use_aux:
        loss_dict['aux_loss'] = {
            'op': torch.nn.CrossEntropyLoss(),
            'weight': 1.0,
            'data_src': ('seg_out', 'seg_label')
        }

    if cfg_train.use_classification:
        loss_dict['classification_loss'] = {
            'op': torch.nn.CrossEntropyLoss(),
            'weight': 1.0,
            'data_src': ('cat_out', 'lane_cls')
        }

    return loss_dict

def get_metric_dict():
    metric_dict = {}

    metric_dict['top1'] = {
        'op': MultiLabelAcc(),
        'data_src': ('cls_out', 'cls_label')
    }

    metric_dict['top2'] = {
        'op': AccTopk(cfg_train.griding_num, 2),
        'data_src': ('cls_out', 'cls_label')
    }

    metric_dict['top3'] = {
        'op':AccTopk(cfg_train.griding_num, 3),
        'data_src': ('cls_out', 'cls_label')
    }

    if cfg_train.use_aux:
        metric_dict['iou'] = {
            'op': Metric_mIoU(cfg_train.num_lanes+1),
            'data_src': ('seg_out', 'seg_label')
        }

    if cfg_train.use_classification:
        metric_dict['cat_acc'] = {
            'op': MultiLabelAcc(),
            'data_src': ('cat_out', 'lane_cls')
        }

    return metric_dict


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)
            
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

        