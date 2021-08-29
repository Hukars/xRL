import torch
import numpy as np
from copy import deepcopy
from offlinerl.algos import BaseOfflineAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.models.check import check_encoder
from offlinerl.utils.models.builders import create_dynamics_model
from offlinerl.utils.torch_utils import soft_sync


class DYNAMICS(BaseOfflineAlgo):   
    def build(self) -> None:
        self.dynamic_model = create_dynamics_model(
                                    self.task.observation_shape,
                                    self.task.action_size,
                                    check_encoder(self.hparams.encoder_factory),
                                    deterministic=self.hparams.deterministic_environment,
                                )
        self.dynamic_model_optim = torch.optim.Adam(self.dynamic_model.parameters(),
                                                     lr=self.hparams.learning_rate,
                                                     weight_decay=0.000075)
            
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        self.dynamic_model_optim.zero_grad()
        loss = self.dynamic_model.compute_error(batch.obs, 
                                                batch.act, 
                                                batch.rew, 
                                                batch.obs_next).mean()
        loss.backward()
        self.dynamic_model_optim.step()
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {"train_loss" : loss.cpu().detach().numpy()}
    
    def train_epoch_end(self, *args):
        pass

    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            dataset = self.val_dataset
        else:
            dataset = self.train_dataset
        val_loader = torch.utils.data.DataLoader(
                            dataset=dataset,
                            batch_size=self.hparams.batch_size,
                            num_workers=4,
                            shuffle = False,
                        )

        return val_loader

    def val_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        with torch.no_grad():
            loss = self.dynamic_model.compute_error(batch.obs, 
                                                    batch.act, 
                                                    batch.rew, 
                                                    batch.obs_next,
                                                    eval=True)                                              
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_epoch_end(self, val_step_outputs):
        pass

    def get_model(self):
        return self.dynamic_model
