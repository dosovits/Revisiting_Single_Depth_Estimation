import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *

import imageio
import time


class depthDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):
        image_name = self.frame.ix[idx, 0]
        depth_name = self.frame.ix[idx, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        sample = {'image': image, 'depth': depth}

        # img_np = np.array(sample["image"])
        # depth_np = np.array(sample["depth"])
        # print("Before transform:", img_np.shape, depth_np.shape)
        # imageio.imwrite("before_img.png", img_np)
        # imageio.imwrite("before_depth.png", depth_np)
        # print(img_np[::100,::100,0])
        # print(depth_np[::100,::100])

        if self.transform:
            sample = self.transform(sample)

        # img_np = np.array(sample["image"])
        # depth_np = np.array(sample["depth"])
        # print("After transform:", img_np.shape, depth_np.shape)
        # imageio.imwrite("after_img.png", np.transpose(img_np, (1,2,0)))
        # imageio.imwrite("after_depth.png", np.transpose(depth_np, (1,2,0)))
        # print(img_np[0,::50,::50])
        # print(depth_np[0,::25,::25])
        # time.sleep(10)

        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(batch_size=64, csv_file='./data/nyu2_train.csv'):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(csv_file=csv_file,
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                           Scale(240),
                                           CenterCrop([304, 228], [304, 228]),
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing
