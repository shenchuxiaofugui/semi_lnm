import torch
import random
import numpy as np
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CenterSpatialCropd,
    RandFlipd,
    RandZoomd,
    RandRotated,
    LoadImaged,
    CropForeground,
)
from trainer.medsam_trainer import ResnetTrainer
import json
from utils.ddp_utils import init_distributed_mode
from utils.Transform import Concat
import os
join = os.path.join
import argparse
import torch.distributed as dist
#设置显卡间通信方式
torch.multiprocessing.set_sharing_strategy('file_system') 
# from torch.optim.swa_utils import AveragedModel #随机权重平均


# fix random seeds for reproducibility
SEED = 2023
def init_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
torch.backends.cudnn.deterministic = False  #在gpu训练时固定随机源
torch.backends.cudnn.benchmark = True   #搜索卷积方式，启动算法前期较慢，后期会快
# torch.autograd.detect_anomaly() # debug的时候启动，用于检查异常



def main(plans_args):
    '''
    设计思路抄自pytorch lighting 和 nnunetv2
    callback 只用在了early stopping, 目前没有其他使用需求
    装饰器只用了静态装饰器和timer
    没有实现梯度累加功能,如果实现记得在DDP模式下取消前几次的梯度同步
    '''
    
    #%% set up transform
    train_transform = Compose([LoadImaged(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"]),
                    EnsureChannelFirstd(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"]),       
                    ScaleIntensityd(["DWI", "T1CE", "T2"]), 
                    CenterSpatialCropd(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"], (200, 200, 12)), 
                    Concat(["DWI", "T1CE", "T2"]),
                    RandFlipd(["image", "roi"], spatial_axis=0, prob=0.5), 
                    RandFlipd(["image", "roi"], spatial_axis=1, prob=0.5), 
                    RandFlipd(["image", "roi"], spatial_axis=2, prob=0.5), 
                    RandZoomd(["image", "roi"], min_zoom=0.7, max_zoom=1.3,  mode=("bilinear", "nearest"), padding_mode="constant", prob=0.7),
                    RandRotated(["image", "roi"], range_z=0.4, mode=("bilinear", "nearest"), padding_mode="zeros", prob=0.5)])
    
    val_transform = Compose([LoadImaged(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"]),
                    EnsureChannelFirstd(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"]), 
                    ScaleIntensityd(["DWI", "T1CE", "T2"]), 
                    CenterSpatialCropd(["DWI", "T1CE", "T2", "DWI_roi", "T1CE_roi", "T2_roi"], (200, 200, 12)), 
                    Concat(["DWI", "T1CE", "T2"]),
 ])
    
    
    #记录一下用了哪些transform，概率是多少
    trans = []
    for i in train_transform.transforms:
        trans.append(i.__class__.__name__)
        if hasattr(i, 'prob'):
            trans.append(i.prob)

    plans_args.config["trans"] = trans    

    # setup data_loader instances
    with open(plans_args.json_path) as f:
        data = json.load(f)
        

    train_ds = CacheDataset(data["train"], transform=train_transform, num_workers=8, cache_rate=0)   
    val_ds = CacheDataset(data["validation"], transform=val_transform, num_workers=8, cache_rate=0)    

    plans_args = init_distributed_mode(plans_args)
    
    # 随机数设置一下
    if not plans_args.ddp:
        init_seeds(seed=SEED)
    else:
        init_seeds(seed=SEED+plans_args.rank)

    # dataloader总得设置一下吧
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if plans_args.ddp else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds) if plans_args.ddp else None
    train_dataloader = DataLoader(train_ds, shuffle=(train_sampler is None),
                    batch_size=plans_args.batch_size, num_workers=8,
                    sampler=train_sampler, pin_memory=True)
    val_dataloader = DataLoader(val_ds, shuffle=(val_sampler is None),
                                batch_size=plans_args.batch_size, num_workers=8,
                                sampler=val_sampler, pin_memory=True)
        

    trainer = ResnetTrainer(
                      plans=plans_args,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                    )

    trainer.train() # 也可以在这里传入model和dataloader
    dist.destroy_process_group()
    


if __name__ == '__main__':
    config = {
        "json_path": "/homes/syli/dataset/LVSI_LNM/dataset.json",
        "resume": None,  #"./saved/model/0807_1536/checkpoint-epoch48.pth"
        "epochs": 200,
        "save_dir":"./saved",
        'network': {"arch": "MultiTaskResNet",
                    "args": None},
        "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 1e-5,
            "weight_decay": 1e-4,
        }},
        "batch_size":25,
        "valid_interval":2,
        "standard": "lvsi_ROCAUC",
        "Metrics": {"Acc":{}, "ROCAUC":{"average":"macro"}},
        "tasks": {"lvsi", "lnm"},
        "early_stopping": 20,
        "criterion": "GHMC",
    }

    plans_args = argparse.Namespace(**config)
    plans_args.config = config
    main(plans_args)