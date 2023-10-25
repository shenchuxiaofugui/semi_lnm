from monai.data import DataLoader, CacheDataset
import torch
from monai.transforms import (
    LoadImaged,
    Compose,
    ScaleIntensityd,
    RandFlip,
    RandRotate,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    RandZoom,
    AddChanneld,
)
from skimage.color import label2rgb
import os
join = os.path.join
import json
#from torch.utils.tensorboard import SummaryWriter

import numpy as np
from model.metric import AccMetric, ROCAUCMetric
import torch


d = {"Acc": AccMetric(), "AUC": ROCAUCMetric()}
for a in d:
    d[a](torch.tensor([0.1, 0.1]), torch.tensor([0, 1]))
    d[a](torch.tensor([0.9, 0.9]), torch.tensor([1, 1]))
for a, b in d.items():
    print(b.aggregate())
    d[a].reset()
for a in d:
    d[a](torch.tensor([0.1, 0.1]), torch.tensor([0, 0]))
    d[a](torch.tensor([0.9, 0.9]), torch.tensor([1, 1]))
for a, b in d.items():
    print(b.aggregate())


# If you have non-default dimension setting, set the dataformats argument.


    
