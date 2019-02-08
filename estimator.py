import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import pdb

import parse
import os

from torchvision import transforms, utils
from demo_transform import *


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


class MonoDepthEstimator:
    def __init__(self):
        self.model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load('./pretrained_model/model_senet'))
        self.model.eval()
        self.init_preprocessor()

    def init_preprocessor(self):
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        self.transform = transforms.Compose([
                            Scale([320, 240]),
                            #CenterCrop([304, 228]),
                            ToTensor(),
                            Normalize(__imagenet_stats['mean'],
                                      __imagenet_stats['std'])
                         ])

    def preprocess(self, image):
        image_torch = self.transform(image).unsqueeze(0)
        return torch.autograd.Variable(image_torch, volatile=True).cuda()
    
    def compute_depth(self, image):
        # Input: image is a PIL image
        # Output: depth is a numpy array
        image_torch = self.preprocess(image)
        print(image_torch.size())
        depth_torch = self.model(image_torch)
        depth = depth_torch.view(depth_torch.size(2),depth_torch.size(3)).data.cpu().numpy()
        return depth

