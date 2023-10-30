import torch
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
from torch.optim.lr_scheduler import CyclicLR
import os
join = os.path.join
from model.model import MultiTaskResNet, MultiTaskDenseNet
#设置显卡间通信方式
torch.multiprocessing.set_sharing_strategy('file_system') 
# from torch.optim.swa_utils import AveragedModel #随机权重平均
# 没有加入callback


# fix random seeds for reproducibility
SEED = 2023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True  #在gpu训练时固定随机源
torch.backends.cudnn.benchmark = True   #搜索卷积方式，启动算法前期较慢，后期会快
# torch.autograd.detect_anomaly() # debug的时候启动，用于检查异常
np.random.seed(SEED)

def main(config):
    
    
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
        
    config["trans"] = trans    

    # setup data_loader instances
    with open(config["json_path"]) as f:
        data = json.load(f)
        

    train_ds = CacheDataset(data["train"], transform=train_transform, num_workers=8, cache_rate=0.1)   
    val_ds = CacheDataset(data["validation"], transform=val_transform, num_workers=8, cache_rate=0.1)    

    is_ddp = config["is_ddp"]
    if is_ddp:
        init_distributed_mode()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        train_dataloader = DataLoader(train_ds,
                        batch_size=config["batch_size"], num_workers=4,
                        sampler=train_sampler, pin_memory=True)
        val_dataloader = DataLoader(val_ds,
                                    batch_size=config["batch_size"], num_workers=4,
                                    sampler=val_sampler, pin_memory=True)
    else:
        train_dataloader = DataLoader(train_ds,
                        batch_size=config["batch_size"], shuffle=True,
                        num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(val_ds, 
                            batch_size=config["batch_size"], shuffle=True,
                            num_workers=4, pin_memory=True)

    # build model architecture, then print to console
    model = MultiTaskResNet(input_channels=3)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer_method = getattr(torch.optim, config["optimizer"]["type"] )
    optimizer = optimizer_method(model.parameters(), 
                        lr=config["optimizer"]["args"]["lr"], weight_decay=config["optimizer"]["args"]["weight_decay"])
    lr_scheduler = CyclicLR(optimizer, config["optimizer"]["args"]["lr"], 
                            config["optimizer"]["args"]["lr"]*3, 48, cycle_momentum=False) # 最小学习率和单个epoch的batch数的3到4倍

    trainer = ResnetTrainer(model, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                      valid_interval=config["valid_interval"],
                      lr_scheduler=lr_scheduler,
                    )

    trainer.train()


if __name__ == '__main__':
    config = {
        "json_path": "/homes/syli/dataset/LVSI_LNM/dataset.json",
        "is_ddp": False,
        "resume": None,  #"./saved/model/0807_1536/checkpoint-epoch48.pth"
        "epochs": 200,
        "save_dir":"./saved",
        'arch': "ResnetTrainer",
        "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 1e-5,
            "weight_decay": 1e-4,
        }},
        "batch_size":25,
        "valid_interval":2,
        "standard": "lvsi_auc",
        "Metrics": {"Acc":{}, "ROCAUC":{"average":"macro"}},
        "tasks": {"lvsi", "lnm"},
    }
    # ddp module的优化meizuo
    main(config)