import torch
from base.base_trainer import BaseTrainer
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from skimage.color import label2rgb
from monai.losses import DiceCELoss
from utils.utils import show_3d_image, judge_log



class SAMTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer, config, 
                 data_loader, valid_data_loader=None, valid_interval=5, lr_scheduler=None):
        super().__init__(model, optimizer, config, data_loader, valid_data_loader, valid_interval)
        self.lr_scheduler = lr_scheduler
        self.criterion = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.train_metrics = DiceMetric(include_background=False, reduction="mean")
        self.valid_metrics = DiceMetric(include_background=False, reduction="mean")
        self.sam_trans = ResizeLongestSide(self.model.image_encoder.img_size)
        self.ori_img_size = (128, 128)
        self.writer_step = 20
        
        if config["resume"] is None:
            checkpoint = torch.load("./work_dir/SAM/medsam_vit_b.pth")
            self.model.load_state_dict(checkpoint)
        
        self.model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图
        
    
    def _train_batch_step(self, batch_data, batch_idx, epoch):
        img2D, gt2D, bbox = (
                batch_data["image"],
                batch_data["label"],
                batch_data["bbox"].to(self.device),
            )
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            show_img = img2D[0,0,...]
            show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
            show_img = label2rgb(gt2D[0,0,...].numpy(), show_img.numpy())
            self.writer.add_image_with_boxes("train_gt", show_img,
                            bbox[0][None,:],epoch* (self.len_epoch // self.writer_step)+(batch_idx // self.writer_step), 
                            dataformats="HWC", thickness=1)
        img2D = img2D.to(self.device)  
        gt2D = gt2D.to(self.device)      
                
        # resize input
        img_1024 = F.interpolate(
            img2D,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
            )            
        # img and prompt encoder
        with torch.no_grad():
            image_embedding = self.model.module.image_encoder(img_1024)
            bbox_torch = self.sam_trans.apply_boxes_torch(bbox, self.ori_img_size)
            sparse_embeddings, dense_embeddings = self.model.module.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
        )
            
        # train decoder
        with torch.cuda.amp.autocast():
            low_res_masks, _ = self.model.module.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=self.model.module.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
            low_res_masks,
            size=self.ori_img_size,
            mode="bilinear",
            align_corners=False,
            )
            loss = self.criterion(ori_res_masks, gt2D)
            # 这里没必要取消梯度缓存，因为encoder根本没算梯度。。。
            # loss = torch.utils.checkpoint.checkpoint(self.criterion,loss)
            
        # train metrics
            med_seg = torch.sigmoid(ori_res_masks)
            med_seg = med_seg > 0.5
            self.train_metrics(y_pred=med_seg, y=gt2D)
            if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
                show_img = img2D[0,0,...]
                show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
                show_med_seg = med_seg.int()
                show_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("train_pred", show_img,
                        bbox[0][None,:],
                        epoch * (self.len_epoch // self.writer_step)+(batch_idx // self.writer_step), 
                        dataformats="HWC", thickness=1)    
        return loss
      
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        epoch_loss = 0
        self.train_metrics.reset()
        self.model.train()
        for batch_idx, batch_data in enumerate(tqdm(self.data_loader)):
            loss = self._train_batch_step(batch_data, batch_idx, epoch)
            epoch_loss = self._train_batch_end(loss, epoch_loss)

        epoch_loss /= batch_idx
        
        # 看情况在哪更新
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
            # current_lr = self.optimizer.learning_rate.numpy()
            # self.writer.add_scalar("Learning Rate", current_lr, step=epoch)
            
        log = {"train_loss": epoch_loss, "train_dice": self.train_metrics.aggregate().item()}

        return log
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch, epoch_loss):
        show_index = epoch % self.writer_step
        img2D, gt2D, bbox = (
                            batch_data["image"].to(self.device),
                            batch_data["label"].to(self.device),
                            batch_data["bbox"].to(self.device),
                        )
        img_1024 = F.interpolate(
            img2D,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False,
            ) 
        
        # img and prompt encoder
        with torch.no_grad():
            image_embedding = self.model.module.image_encoder(img_1024)
            bbox_torch = self.sam_trans.apply_boxes_torch(bbox, self.ori_img_size)
            sparse_embeddings, dense_embeddings = self.model.module.prompt_encoder(
            points=None,
            boxes=bbox_torch,
            masks=None,
            )
            

            low_res_masks, _ = self.model.module.mask_decoder(
                image_embeddings=image_embedding, # (B, 256, 64, 64)
                image_pe=self.model.module.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
                multimask_output=False,
            )
            ori_res_masks = F.interpolate(
            low_res_masks,
            size=self.ori_img_size,
            mode="bilinear",
            align_corners=False,
            )
            loss = self.criterion(ori_res_masks, gt2D)
            
        # validation metrics
            med_seg = torch.sigmoid(ori_res_masks)
            med_seg = med_seg > 0.5
            self.valid_metrics(y_pred=med_seg, y=gt2D)
            epoch_loss += loss.item()
            if judge_log(self.is_ddp) and (batch_idx+show_index) % self.writer_step == 0:
                show_img = img2D[0,0,...]
                show_img = (show_img - show_img.min()) / (show_img.max() - show_img.min())
                show_med_seg = med_seg.int()
                show_pred_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("test_pred", show_pred_img,
                                bbox[0][None,:], #batch changdu
                        epoch* (self.len_epoch // self.writer_step)+((batch_idx+show_index) // self.writer_step),
                                dataformats="HWC", thickness=1)

                show_med_seg = gt2D.int()
                show_gt_img = label2rgb(show_med_seg[0,0,...].cpu().numpy(), show_img.cpu().numpy())
                self.writer.add_image_with_boxes("test_gt", show_gt_img,
                                bbox[0][None,:],
                        epoch* (self.len_epoch // self.writer_step)+((batch_idx+show_index) // self.writer_step),
                                dataformats="HWC", thickness=1)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        epoch_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(self.valid_data_loader)):
            epoch_loss = self._valid_batch_step(self, batch_data, batch_idx, epoch, epoch_loss)
            
        # 按lighting再改改        
        epoch_loss /= batch_idx
        log = {"loss": epoch_loss, "dice": self.valid_metrics.aggregate().item()}
        return log
    


class ResnetTrainer(BaseTrainer):
    def __init__(self, plans, data_loader, valid_data_loader=None):
        # model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图
        super().__init__(plans, data_loader, valid_data_loader)
        self.writer_step = 5
        # model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图,还未支持3.11

    def _load_and_visualize_input(self, batch_data, batch_idx, epoch, mode):
        img = batch_data["image"]
        if judge_log(self.is_ddp) and batch_idx % self.writer_step == 0:
            for i, modal in enumerate(["DWI", "T1CE", "T2"]):
                step = epoch*self.len_epoch+(batch_idx // self.writer_step)
                show_3d_image(img[:,i,...].numpy(), batch_data["roi"][:,i,...].numpy(), self.writer, modal, step, mode)
        img = img.to(self.device)
        label1 = batch_data["label"][0].to(torch.float).to(self.device)
        label2 = batch_data["label"][1].to(torch.float).to(self.device)
        return img, label1, label2

    def _train_batch_step(self, batch_data, batch_idx, epoch):
        img, label1, label2 = self._load_and_visualize_input(batch_data, batch_idx, epoch, "train")
        pred1, pred2 = self.model(img)
        pred1, pred2 = torch.squeeze(pred1), torch.squeeze(pred2)
        label_weight1 = torch.where(label1 == 1, torch.tensor(0.738), torch.tensor(0.262))
        label_weight2 = torch.where(label2 == 1, torch.tensor(0.829), torch.tensor(0.171))
        loss = self.criterion(pred1, label1, label_weight1) + self.criterion(pred2, label2, label_weight2)
        # updata metrics
        for pred, label, task in zip([pred1, pred2], [label1, label2], ["lvsi", "lnm"]):
            for metric in self.plans.Metrics:
                self.metrics[task+"_"+metric](y_pred=pred, y=label)
        return loss
        
    def _valid_batch_step(self, batch_data, batch_idx, epoch):
        img, label1, label2 = self._load_and_visualize_input(batch_data, batch_idx, epoch, "valid")
        with torch.no_grad():
            pred1, pred2 = self.model(img)
            # calulate loss
            pred1, pred2 = torch.squeeze(pred1), torch.squeeze(pred2)
            label_weight1 = torch.where(label1 == 1, torch.tensor(0.738), torch.tensor(0.262))
            label_weight2 = torch.where(label2 == 1, torch.tensor(0.829), torch.tensor(0.171))
            loss = self.criterion(pred1, label1, label_weight1) + self.criterion(pred2, label2, label_weight2)
            # loss = self.criterion(pred1, label1) + self.criterion(pred2, label2)
            # updata metrics
            for pred, label, task in zip([pred1, pred2], [label1, label2], ["lvsi", "lnm"]):
                for metric in self.plans.Metrics:
                    self.metrics[task+"_"+metric](y_pred=pred, y=label)
        return loss.item()