import copy
import random
import torch
import torch.nn as nn
import numpy as np
from offlinerl.utils.trainer_utils import select_algo
import pytorch_lightning as pl
from omegaconf import OmegaConf
from typing import Union, Dict
from offlinerl.utils.conf_utils import get_algo_conf, get_default_hparams


algo_by_val_loss = ["DYNAMICS", "ENSEMBLEDYNAMICS" ]


class BaseTrainer:
    def __init__(self, 
                 task,
                 algo_name,
                 seed=0,
                 need_log=True,
                 **kwargs) -> None:        
        pl.seed_everything(seed)
        self._algo_name = algo_name
        self._algo_cls = select_algo(algo_name)
        self._algo_conf = get_algo_conf(algo_name)
        self._task = task
        self.need_log = need_log
        self.init_kwargs = kwargs
        
        self.trainer_args = self.prepare_trainer_args(kwargs)
        
    def prepare_trainer_args(self, trainer_args):
        trainer_args["reload_dataloaders_every_epoch"] = True
        trainer_args["max_epochs"] = 100
        trainer_args["num_sanity_val_steps"] = 5
        trainer_args["weights_summary"] = None
        if "limit_train_batches" not in trainer_args.keys():
            trainer_args["limit_train_batches"] = 1000
        if ("logger" not in trainer_args.keys() or "callbacks" not in trainer_args.keys()) and self.need_log:
            raise ValueError("Please define the pl.callbacks.ModelCheckpoint instance and logger instance"
                             "when initialize the BaseTrainer.")
        return trainer_args
    
    def algo_conf(self, conf_level="total"):
        if conf_level == "default":
            pass
        elif conf_level == "search_space":
            pass
        
        return copy.deepcopy(self._algo_conf)
        
    def init_task(self): 
        task = self._task 
        
        return task
        
    def init_algo(
        self, 
        conf: Union[dict, OmegaConf] = None, 
    ):
        hparams = get_default_hparams(self._algo_conf)
        if conf:
            #判断dict是否需要转conf
            hparams = hparams.update(get_default_hparams(conf))
        algo = self._algo_cls(hparams, self.task)
        
        return algo
    
    def init_trainer(self):
        trainer = pl.Trainer(**self.trainer_args)
        trainer.init_kwargs = self.init_kwargs
        
        return trainer
    
    def on_before_train(self, conf=None):
        self.task = self.init_task()
        self.algo = self.init_algo(conf)
        self.trainer = self.init_trainer()
        
    def train(self, conf: Union[dict, OmegaConf] = None, pretrain_model_path: str = None, addition_models: Dict[str, nn.Module] = None):
        self.on_before_train(conf)
        if addition_models is not None:
            self.algo.addition_models = addition_models
        if pretrain_model_path is not None:
            #self.algo.load_from_checkpoint(pretrain_model_path)
            self.algo.load_state_dict(torch.load(pretrain_model_path)['state_dict'])
        self.trainer.fit(self.algo)
        
        return self.algo
