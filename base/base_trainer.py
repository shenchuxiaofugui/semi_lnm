import torch
from abc import abstractmethod
from numpy import inf
from tensorboardX import SummaryWriter
from utils.utils import EarlyStopping
import logging
import time
import os
from pathlib import Path
join = os.path.join
from torch.nn.parallel import DistributedDataParallel as DDP
from models import metric, model
from utils.utils import judge_log
from models.lr_scheduler import PolyLRScheduler
from tqdm import tqdm
from models import loss
import torch.distributed as dist



class BaseTrainer(object):
    """
    Base class for all trainers
    没多大可能动的设置放在此处
    """
    def __init__(self, plans, data_loader, valid_data_loader=None,):
        self.data_loader = data_loader # 训练数据
        self.valid_data_loader = valid_data_loader  # 验证数据
        self.do_validation = self.valid_data_loader is not None  # 是否做验证
        self.valid_interval = plans.valid_interval # 每几个epoch做一次验证
        self.plans = plans  # 一些超参的设置
        self.device = "cuda"
        self.scaler = torch.cuda.amp.GradScaler()  #采用amp
        self.start_epoch = 0 # 开始的epoch，有checkpoint会更新
        self.mnt_best = 0 # 最优的验证集metric
        self.epochs = plans.epochs # 最大训练轮次
        self.len_epoch = len(self.data_loader) # 一个epoch有多少个batch
        self.early_stopping = EarlyStopping(plans.early_stopping) # 设置早停

        self._build_model()    # build model architecture, then print to console

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        self.optimizer = getattr(torch.optim, plans.optimizer["type"])(self.model.parameters(), **plans.optimizer["args"])
        self.lr_scheduler = PolyLRScheduler(self.optimizer, plans.optimizer["args"]["lr"], self.epochs)

        # loss
         
        self.criterion = getattr(loss, plans.criterion)()

        # configuration to monitor model performance and save best
        self._init_logger()
        
        # 加载一些要用的metrics
        self._load_metrics()

        # 加载checkpoint
        if plans.resume is not None:
            self._resume_checkpoint(plans.resume)
            
        # model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图

        # setup visualization writer instance                
        self.writer = SummaryWriter(log_dir=self.logger_path)
        
        # DDP加载
        self.is_ddp = plans.ddp
        if self.is_ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.device])

    def _build_model(self):
    # build model architecture, then print to console
        network_class = getattr(model, self.plans.network['arch'])
        if self.plans.network["args"] is not None:
            network = network_class(**self.plans.network["args"])
        else:
            network = network_class()
        self.model = network.to(self.device)

    def _init_logger(self):
        # configuration to monitor model performance and save best
        now_time = self._get_time()
        self.model_path = join(self.plans.save_dir, "model", now_time)
        self.logger_path = join(self.plans.save_dir, "log", now_time)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.logger_path, exist_ok=True)
        self.checkpoint_dir = Path(self.model_path)
        self.logger = self._set_logger(self.logger_path)

    def _load_metrics(self):
        metrics_dict = self.plans.Metrics
        self.metrics = {}
        for key in metrics_dict:
            one_metric = getattr(metric, key+"Metric")
            if hasattr(self.plans, 'tasks'):
                for task in self.plans.tasks:
                    self.metrics.update({task+"_"+key: one_metric(**metrics_dict[key])})
            else:
                self.metrics.update({key: one_metric(**metrics_dict[key])})

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        epoch_loss = self._epoch_start(epoch, "train")
        # 每一batch要做的事
        for batch_idx, batch_data in enumerate(tqdm(self.data_loader)):
            loss = self._train_batch_step(batch_data, batch_idx, epoch)
            epoch_loss += self._train_batch_end(loss)

        log = self._train_epoch_end(epoch_loss, batch_idx)

        return log
    
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        epoch_loss = self._epoch_start(epoch, "valid")

        for batch_idx, batch_data in enumerate(tqdm(self.valid_data_loader)):
            epoch_loss += self._valid_batch_step(batch_data, batch_idx, epoch)
            
        # epoch结束    
        log = self._train_epoch_end(epoch_loss, batch_idx)

        return log

    @abstractmethod
    def _train_batch_step(self, batch_data, batch_idx, epoch):
    # Training logic for an epoch
        raise NotImplementedError
    
    def _valid_batch_step(self, batch_data, batch_idx, epoch):
        raise NotImplementedError

    @abstractmethod
    def _load_and_visualization_input(self):
        raise NotImplementedError
    
    def _update_metrics(self):
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        torch.cuda.empty_cache() # 训练刚开始清一下缓存
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 每一epoch要实现的事
            log = {'epoch': epoch}
            # save train informations into log dict
            if self.is_ddp:
                self.data_loader.sampler.set_epoch(epoch)
                self.valid_data_loader.sampler.set_epoch(epoch) # 记得每个epoch打乱一下sampler
                dist.barrier()
            result = self._train_epoch(epoch)
            if self.is_ddp:
                dist.barrier()
            log.update(**{'train_'+k : v for k, v in result.items()})

            # save validtion informations into log dict
            if self.do_validation and (epoch + 1) % self.valid_interval == 0:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_'+k : v for k, v in val_log.items()})
            
            self._on_epoch_end(log, epoch)
            # early stop
            if self.early_stopping.early_stop:
                break

    def _on_epoch_end(self, log, epoch):
        # print logged informations to the screen
        for key, value in log.items():
            if key == "epoch":
                continue
            if judge_log(self.is_ddp): 
                metric_key = key.split("_")[-1]
                self.writer.add_scalars(metric_key, {key: value}, epoch)
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best = False
        if "val_" + self.plans.standard in log.keys():
            if self.mnt_best < log["val_" + self.plans.standard]:
                self.mnt_best = log["val_" + self.plans.standard]
                best = True
        
        if judge_log(self.is_ddp):
            self._save_checkpoint(epoch, save_best=best)
        
        self.early_stopping(log["train_loss"])
        if self.early_stopping.early_stop:
            self.logger.info("Early stopping")
            self.logger.info("Best val_{}: {}".format(self.plans.standard, self.mnt_best))

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.is_ddp:
            model_state_dict = self.model.module.state_dict()
        else: 
            model_state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.plans.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        old_file = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch-5))
        if os.path.exists(old_file):
            os.remove(old_file)
                
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['network']['arch'] != self.plans.network['arch']:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        # my_dic_keys = list(checkpoint['state_dict'].keys())
        # for key in my_dic_keys:
        #     checkpoint['state_dict'][key.replace("module.", "")] = checkpoint['state_dict'].pop(key)
        self.model.eval()
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.plans.optimizer['type']:
            print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        self.model_path = os.path.dirname(resume_path)
        self.logger_path = join(self.model_path.replace("model", "log"))

    @staticmethod    
    def _get_time():
        t = time.localtime()
        now = time.strftime("%m%d_", t) + str(t.tm_hour+8)+str(t.tm_min)
        return now
        
    @staticmethod     
    def _set_logger(logger_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(join(logger_path, "log.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger
    
    def _train_batch_end(self, batch_loss):
        self.optimizer.zero_grad()
        self.scaler.scale(batch_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        return batch_loss.item()
    
    def _train_epoch_end(self, epoch_loss, batch_idx):
        epoch_loss /= batch_idx  
        log = {"loss": epoch_loss}
        for key, value in self.metrics.items():
            log[key] = value.aggregate()
        return log
    
    def _epoch_start(self, epoch, mode):
        for metric in self.metrics:
            self.metrics[metric].reset()
        if mode == "train":
            self.model.train()
            self.lr_scheduler.step()
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch) #可以用callback实现，但是没有必要
        elif mode == "valid":
            self.model.eval()
        return 0