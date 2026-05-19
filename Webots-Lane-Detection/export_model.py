import torch
import os
from model.model import UFLDNet
from configs import cfg_common, cfg_demo

cls_num_per_lane = 56

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

example = torch.zeros(1, 3, 288, 800).cuda()
traced = torch.jit.trace(model, example, strict=False)
traced.save(f"{os.path.splitext(cfg_demo.model_path)[0]}_traced.pt")