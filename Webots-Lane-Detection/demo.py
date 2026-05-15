import torch
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import scipy.special
from tqdm import tqdm
import numpy as np

from model.model import UFLDNet
from data.dataset import DemoDataset
from configs import cfg_common, cfg_demo

def main():
    # number of row anchors
    if cfg_demo.dataset == 'Carla':
        cls_num_per_lane = 56
        img_w, img_h = 1280, 720
        row_anchor = cfg_common.carla_row_anchor
    else:
        raise NotImplementedError

    # define model
    
    model = UFLDNet(
        pretrained=False,
        backbone=cfg_demo.backbone,
        cls_dim=(cfg_demo.griding_num + 1, cls_num_per_lane, cfg_demo.num_lanes),
        cat_dim=(cfg_demo.num_lanes, cfg_demo.num_cls),
        use_aux=False, # we dont need auxiliary segmentation in testing
        use_classification=cfg_demo.use_classification
    )
    model.cuda()

    # load model weights
    # If your model was trained with torch.nn.DataParallel or DistributedDataParallel, the state_dict keys are prefixed with 'module.'
    # If you’re now loading into a single-GPU or CPU model, the keys need to match exactly. So this loop removes the 'module.' prefix from the keys.
    state_dict = torch.load(cfg_demo.model_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
            
    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()

    # transform input image
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset = DemoDataset(cfg_demo.img_folder, img_transform=img_transforms)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for batch_idx, (imgs, img_paths) in enumerate(tqdm(dataloader)):
        imgs = imgs.cuda()
        # predict
        with torch.no_grad():
            out = model(imgs) # (batch_size, num_gridding, num_cls_per_lane, num_of_lanes)

        detection = out['det']

        col_sample = np.linspace(0, 800 - 1, cfg_demo.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = detection[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :] # flips rows
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0) # removes the last class, which is often reserved for no lane / background.
        idx = np.arange(cfg_demo.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0) # expectation / avg idx
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg_demo.griding_num] = 0
        out_j = loc

        vis = cv2.imread(img_paths[0])
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(vis,ppp,5,(0,255,0),-1)
        
        img_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
        cv2.imwrite(os.path.join(cfg_demo.out_folder, img_name + '.png'), vis)

if __name__ == "__main__":
    main()