import torch
from .backbone import ResNet
import numpy as np

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class UFLDNet(torch.nn.Module):
    def __init__(self, pretrained, backbone, cls_dim, cat_dim, use_aux, use_classification):
        super(UFLDNet, self).__init__()

        # this is the dimension of the model output used for group classification. (width, height, channel)
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        self.cat_dim = cat_dim # (num_of_lanes, num_classification)
        self.use_aux = use_aux
        self.use_classification = use_classification
        self.total_dim = np.prod(cls_dim)

        # input : nchw,
        # output: (w+1) * sample_rows * 4 (num_of_lanes) 
        
        ### Res blocks ###
        self.resnet = ResNet(backbone, pretrained=pretrained)

        ### Auxiliary segmentation ###
        if self.use_aux:
            self.aux_header2 = torch.nn.Sequential(
                conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
                conv_bn_relu(128, 128, 3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                conv_bn_relu(384, 256, 3,padding=2,dilation=2),
                conv_bn_relu(256, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=2,dilation=2),
                conv_bn_relu(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, cls_dim[-1] + 1,1)
                # output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)

        ### Additional convolutional layer to further processes features ###
        if backbone in ['34','18']:
            self.conv = torch.nn.Conv2d(512, 8, 1)
        else: 
            self.conv =torch.nn.Conv2d(2048, 8, 1)

        ### Detection (Group classification) ###
        self.det = torch.nn.Sequential(
            torch.nn.Linear(1800, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )
        initialize_weights(self.det)
        
        ### Classification ###
        if self.use_classification:
            self.category = torch.nn.Sequential(
                torch.nn.Linear(1800, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, np.prod(self.cat_dim))
            )

    def forward(self, x):
        # n c h w - > n 2048 sh sw
        # -> n 2048
        x2, x3, fea = self.resnet(x)

        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
            x4 = self.aux_header4(fea)
            x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
            aux_seg = torch.cat([x2,x3,x4],dim=1)
            aux_seg = self.aux_combine(aux_seg)

        # fea is torch dim = (8, 9, 25), since input image is (288, 800)
        # flatten
        fea = self.conv(fea).view(-1, 1800) # (batch_size, input vector dim)

        detection = self.det(fea).view(-1, *self.cls_dim)

        output = {}
        output['det'] = detection

        if self.use_classification:
            category = self.category(fea).view(-1, *self.cat_dim)        
            output['cat'] = category

        if self.use_aux:
            output['aux'] = aux_seg
        
        return output

def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
        
def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)