import torch
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, random_split

import data.mytransforms as mytransforms
from .dataset import LaneClsDataset

def get_data_loader(batch_size, data_root, griding_num, dataset, use_aux, num_lanes, row_anchor):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])

    if dataset == 'Carla':
        full_dataset = LaneClsDataset(data_root,
                                       os.path.join(data_root, 'train_gt.txt'),
                                       img_transform=img_transform,
                                       target_transform=target_transform,
                                       simu_transform=None,
                                       griding_num=griding_num, 
                                       row_anchor=row_anchor,
                                       segment_transform=segment_transform,
                                       use_aux=use_aux, 
                                       num_lanes=num_lanes)
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    # Define split sizes
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f"Number of training data: {len(train_dataset)}")
    print(f"Number of testing data: {len(test_dataset)}")

    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root, dataset):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset == 'CULane':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'list/test.txt'),img_transform = img_transforms)
        cls_num_per_lane = 18
    elif dataset == 'Tusimple':
        test_dataset = LaneTestDataset(data_root,os.path.join(data_root, 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56

    sampler = torch.utils.data.SequentialSampler(test_dataset)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler = sampler, num_workers=4)
    return loader