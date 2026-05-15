import torch
import os
import numpy as np
import yaml
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.model import UFLDNet
from data.dataloader import get_data_loader
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from configs import cfg_common, cfg_train

def inference(net, data_label, use_aux, use_classification):
    img = data_label["img"]
    img = img.cuda()
    cls_label = data_label["cls_label"]
    cls_label = cls_label.long().cuda()
    if use_aux:
        seg_label = data_label["seg_label"]
        seg_label = seg_label.long().cuda()
    if use_classification:
        lane_cls = data_label["lane_cls"]
        lane_cls = lane_cls.long().cuda()  
    
    output = net(img)
    results = {'cls_out': output['det'], 'cls_label': cls_label}
    if use_aux:
        results['seg_out'] = output['aux']
        results['seg_label'] = seg_label
    if use_classification:
        results['cat_out'] = output['cat']
        results['lane_cls'] = lane_cls

    return results


def resolve_val_data(results, use_aux, use_classification):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    if use_classification:
        results['cat_out'] = torch.argmax(results['cat_out'], dim=1)
    return results

def calc_loss(loss_dict, results, logger, global_step, split):
    loss = 0

    for k, v in loss_dict.items():
        data_src = v['data_src']
        datas = [results[src] for src in data_src]

        loss_cur = v['op'](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(f'{split}_loss/'+k, loss_cur, global_step)

        loss += loss_cur * v['weight']

    return loss

def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux, use_classification):
    net.train()

    progress_bar = tqdm(data_loader)
    t_data_0 = time.time()

    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux, use_classification)

        loss = calc_loss(loss_dict, results, logger, global_step, 'train')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux, use_classification)

        update_metrics(metric_dict, results)

        if global_step % 20 == 0:
            for k, v in metric_dict.items():
                me_name = k
                me_op = v['op']
                logger.add_scalar('train_metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        kwargs = {}
        for k, v in metric_dict.items():
            me_name = k
            me_op = v['op']
            kwargs[me_name] = '%.3f' % me_op.get()
        
        progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                **kwargs)
        t_data_0 = time.time()

def test(net, data_loader, loss_dict, logger, epoch, metric_dict, use_aux, use_classification):
    net.eval()

    progress_bar = tqdm(data_loader)
    t_data_0 = time.time()

    with torch.no_grad():
        for b_idx, data_label in enumerate(progress_bar):
            t_data_1 = time.time()
            reset_metrics(metric_dict)
            global_step = epoch * len(data_loader) + b_idx

            t_net_0 = time.time()
            results = inference(net, data_label, use_aux, use_classification)
            t_net_1 = time.time()

            loss = calc_loss(loss_dict, results, logger, global_step, 'test')

            results = resolve_val_data(results, use_aux, use_classification)

            update_metrics(metric_dict, results)

            if global_step % 20 == 0:
                for k, v in metric_dict.items():
                    me_name = k
                    me_op = v['op']
                    logger.add_scalar('test_metric/' + me_name, me_op.get(), global_step=global_step)

            kwargs = {}
            for k, v in metric_dict.items():
                me_name = k
                me_op = v['op']
                kwargs[me_name] = '%.3f' % me_op.get()
            
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
            t_data_0 = time.time()

def save_model(net, optimizer, epoch, save_path):
    model_state_dict = net.state_dict()
    state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
    assert os.path.exists(save_path)
    model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
    torch.save(state, model_path)
        
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader, cls_num_per_lane = get_data_loader(cfg_train.batch_size, cfg_train.data_root, cfg_train.griding_num, cfg_train.dataset, cfg_train.use_aux, cfg_train.num_lanes, cfg_common.carla_row_anchor)

    net = UFLDNet(
        pretrained=True, 
        backbone=cfg_train.backbone, 
        cls_dim=(cfg_train.griding_num + 1, cls_num_per_lane, cfg_train.num_lanes), 
        cat_dim=(cfg_train.num_lanes, cfg_train.num_cls), 
        use_aux=cfg_train.use_aux, 
        use_classification=cfg_train.use_classification
    )
    net.cuda()

    optimizer = get_optimizer(net)

    if cfg_train.finetune is not None:
        print('finetune from ', cfg_train.finetune)
        state_all = torch.load(cfg_train.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg_train.resume is not None:
        print('==> Resume model from ' + cfg_train.resume)
        resume_dict = torch.load(cfg_train.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg_train.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, len(train_loader))
    metric_dict = get_metric_dict()
    loss_dict = get_loss_dict()

    logger = SummaryWriter()
    
    os.makedirs(cfg_train.save_path)

    for epoch in range(resume_epoch, cfg_train.epoch):
        print(f"Starting epoch {epoch}")
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg_train.use_aux, cfg_train.use_classification)
        test(net, test_loader, loss_dict, logger, epoch, metric_dict, cfg_train.use_aux, cfg_train.use_classification)

    save_model(net, optimizer, epoch , cfg_train.save_path)
    
    logger.close()