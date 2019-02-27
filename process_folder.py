import argparse
import torch
import torch.nn.parallel

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("gray")

import parse
import os

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

def get_files(path, in_template = "{}_color.png", out_template = "{}_pred.png"):
    in_files = []
    out_files = []
    for f in os.listdir(path):
        match = parse.parse(in_template, f)
        if match:
            in_files.append(os.path.join(path, f))
            out_files.append(os.path.join(path, out_template.format(match[0])))
    return in_files, out_files



def main():
    # model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_resnet'))
    # model.load_state_dict(torch.load('./trained_models/01_resnet_ft_on_mp3d/checkpoint_0.pth.tar')['state_dict'])
    model.load_state_dict(torch.load('./trained_models/02_resnet_ft_on_mp3d_lr_4x_smaller/checkpoint_1.pth.tar')['state_dict'])
    # model.load_state_dict(torch.load('./trained_models/resnet_nyu/checkpoint.pth.tar')['state_dict'])
    model.eval()

    path = "/home/adosovit/work/toolboxes/2019/navigation-benchmark/3rdparty/minos/screenshots"
    in_files, out_files = get_files(path)

    for in_file, out_file in zip(in_files, out_files):
        # print(in_file, out_file)
        img_loader = loaddata.readNyu2(in_file)
        test(img_loader, model, out_file = out_file)



def test(nyu2_loader, model, out_file = "data/demo/out.png"):
    for i, image in enumerate(nyu2_loader):
        print(image.size())
        image = image[:,:3,:,:]
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)
        out_np = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        out_np = (out_np / np.max(out_np) * 255.).astype(np.uint8)

        matplotlib.image.imsave(out_file, out_np)

if __name__ == '__main__':
    main()
