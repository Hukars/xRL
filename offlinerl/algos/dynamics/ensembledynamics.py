import torch
import numpy as np
from copy import deepcopy
from offlinerl.algos import BaseOfflineAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.models.check import check_encoder
from offlinerl.utils.models.builders import create_ensemble_dynamics_model
from offlinerl.utils.torch_utils import soft_sync


class ENSEMBLEDYNAMICS(BaseOfflineAlgo):   
    def build(self) -> None:
        self.dynamic_models = create_ensemble_dynamics_model(
                                    self.task.observation_shape,
                                    self.task.action_size,
                                    check_encoder(self.hparams.encoder_factory),
                                    self.hparams.n_init_ensembles,
                                    deterministic=self.hparams.deterministic_environment,
                                )

        self.best_dynamic_models = create_ensemble_dynamics_model(
                                    self.task.observation_shape,
                                    self.task.action_size,
                                    check_encoder(self.hparams.encoder_factory),
                                    self.hparams.n_init_ensembles,
                                    deterministic=self.hparams.deterministic_environment,
                                )
        self.val_losses = [float('inf') for i in range(len(self.dynamic_models.models))]
        self.update_flag = [False] * len(self.dynamic_models.models)
        self.dynamic_models_optim = torch.optim.Adam(self.dynamic_models.parameters(),
                                                     lr=self.hparams.learning_rate,
                                                     weight_decay=0.000075)
            
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        self.dynamic_models_optim.zero_grad()
        loss = self.dynamic_models.compute_error(batch.obs, 
                                                 batch.act, 
                                                 batch.rew, 
                                                 batch.obs_next)
        loss.backward()
        self.dynamic_models_optim.step()
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
            loss, loss_list = self.dynamic_models.compute_error(batch.obs, 
                                                                batch.act, 
                                                                batch.rew, 
                                                                batch.obs_next,
                                                                eval=True)
            best_loss, best_loss_list = self.best_dynamic_models.compute_error(batch.obs, 
                                                                               batch.act, 
                                                                               batch.rew, 
                                                                               batch.obs_next,
                                                                               eval=True)                                               
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        print('loss_list', loss_list)
        print('best_loss_list', best_loss_list)
        return loss_list
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss_list = np.mean(np.array(val_step_outputs), axis=0)
        for i, (new_loss, old_loss) in enumerate(zip(avg_loss_list, self.val_losses)):
            if new_loss < old_loss:
                self.update_flag[i] = True
                self.val_losses[i] = new_loss
            else:
                self.update_flag[i] = False
        print(self.update_flag)
        for i, (flag, model, targ_model) in enumerate(zip(self.update_flag, 
                                                          self.dynamic_models.models,
                                                          self.best_dynamic_models.models)):
            if flag:
                targ_model.load_state_dict(deepcopy(model.state_dict()))
        
        indices = self._select_best_indexes(self.val_losses, self.hparams.n_init_ensembles)
        print(indices)
        self.best_dynamic_models.best_model_indices.data = torch.tensor(indices, device=self.device)
                
    def get_model(self):
        return self.best_dynamic_models
    
    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes
