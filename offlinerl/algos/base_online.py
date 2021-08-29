from abc import abstractmethod

import torch
import pytorch_lightning as pl
from loguru import logger
from offlinerl.utils.dataset import Batch
from offlinerl.evaluation import test_on_real_env
from offlinerl.utils.dataset import SampleBatch, make_buffer_dataset


class BaseOnlineAlgo(pl.LightningModule):
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
    def train_step_end(self,):
        pass
    
    def training_step(self, batch, batch_idx):
        logs = self.train_step(Batch(batch))
        return logs
    
    def training_step_end(self, training_step_outputs):
        log = self.train_step_end(training_step_outputs)
        return log

    def train_dataloader(self):
        """Pytorch lightning use this function to load data, but I make it unused in the implementation in sac.
           Instead, in every train step, I just sample a batch from a replay_buffer.
        """
        
        train_loader = torch.utils.data.DataLoader(
                        dataset=torch.empty((self.trainer.limit_train_batches)),
                        batch_size=1,
                        num_workers=32,
                        shuffle = False,
                    )

        return train_loader
    
    def configure_optimizers(self):
        pass
  
    def on_train_start(self):
        if "max_epochs" in self.hparams.keys():
            assert isinstance(self.hparams.max_epochs, int) and self.hparams.max_epochs > 0
            self.trainer.max_epochs = self.hparams.max_epochs           
        if not self.enable_trainer_logger:
            pass
        # set default data type
        torch.set_default_dtype(torch.float32)
        
    def prepare_data(self):   
        if self.hparams.need_replay_buffer:
            self.train_buffer = make_buffer_dataset('re', self.hparams.buffer_size)
        else:
            self.train_buffer = SampleBatch()
