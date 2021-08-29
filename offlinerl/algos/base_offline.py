from abc import abstractmethod
import torch
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from offlinerl.utils.dataset import Batch
from offlinerl.evaluation import test_on_real_env
from offlinerl.utils.dataset import make_buffer_dataset


class BaseOfflineAlgo(pl.LightningModule):
    def __init__(self, hparams, task, enable_trainer_logger=True):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = task
        self.addition_models = None
        self.enable_trainer_logger = enable_trainer_logger
        self.automatic_optimization = False
        self.build()
    
    @abstractmethod
    def build(self):
        pass
    
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def train_step(self,):
        pass
    
    @abstractmethod
    def val_step(self,):
        pass

    def train_epoch_end(self, *args):
        res = test_on_real_env(self.get_model(), self.task["env"], self.device, max_env_steps=self.hparams.max_ep_length if 'max_ep_length' in self.hparams.keys() else 1000)
        logger.info(f"result: {res} ")
        self.log("val_mean_reward", res['Reward_Mean_Env'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("Length_Mean_Env", res['Length_Mean_Env'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("Success_Rate", res['Success_Rate'], on_step=False, on_epoch=True, prog_bar=True)
        return res
    
    def training_step(self, batch, batch_idx):
        logs = self.train_step(Batch(batch))
        return logs
    
    def training_epoch_end(self, training_step_outputs):
        self.train_epoch_end(training_step_outputs)
    
    def configure_optimizers(self):
        pass
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
                            dataset=self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=32,
                            shuffle = True,
                        )

        return train_loader
    
    def on_train_start(self):
        if "max_epochs" in self.hparams.keys():
            assert isinstance(self.hparams.max_epochs, int) and self.hparams.max_epochs > 0
            self.trainer.max_epochs = self.hparams.max_epochs
            
        if "steps_per_epoch" in self.hparams.keys():
            assert isinstance(self.hparams.steps_per_epoch, int) and self.hparams.steps_per_epoch > 0
            self.trainer.limit_train_batches = self.hparams.steps_per_epoch
            
        if not self.enable_trainer_logger:
            pass
        # set default data type
        torch.set_default_dtype(torch.float32)
        
    def prepare_data(self):        
        if "val_ratio" in self.hparams.keys():
            dataset_indices = np.random.permutation(len(self.task["train_buffer"]))
            self.train_dataset = make_buffer_dataset("b", self.task["train_buffer"][dataset_indices[:int(len(self.task["train_buffer"])*self.hparams.val_ratio)]])
            self.val_dataset = make_buffer_dataset("b", self.task["train_buffer"][dataset_indices[int(len(self.task["train_buffer"])*self.hparams.val_ratio):]])
        else:
            self.train_dataset = make_buffer_dataset("b", self.task["train_buffer"])
