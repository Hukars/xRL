import torch

from offlinerl.algos import BaseOfflineAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.models.check import check_encoder
from offlinerl.utils.models.builders import create_deterministic_regressor, create_squashed_normal_policy


class BC(BaseOfflineAlgo):
    def build(self) -> None:
        self.actor = create_deterministic_regressor(
                            self.task.observation_shape, 
                            self.task.action_size, 
                            check_encoder(self.hparams.encoder_factory),
                            )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.learning_rate)

    # In BC algorithm, we need validation set
    def val_dataloader(self):
        if hasattr(self, "val_dataset"):
            dataset = self.val_dataset
        else:
            dataset = self.train_dataset
        val_loader = torch.utils.data.DataLoader(
                            dataset=dataset,
                            batch_size=len(dataset),
                            num_workers=4,
                            shuffle = False,
                        )

        return val_loader        
    
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        self.actor_optim.zero_grad()
        loss = self.actor.compute_error(batch.obs, batch.act)
        loss.backward()
        self.actor_optim.step()
        
        return {"train_loss" : loss.cpu().detach().numpy().item()}
    
    def val_step(self,):
        pass
    
    def validation_step(self, batch, batch_idx):
        batch = to_torch(batch, torch.float, device=self.device)
        with torch.no_grad():
            loss = self.actor.compute_error(batch.obs, batch.act)                                         
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss" : loss.cpu().detach().numpy().item()}
     
    def get_model(self):
        return self.actor
    

class BCP(BaseOfflineAlgo):
    def build(self):
        # build bc policy model
        self.target_entropy_temp = -self.task.action_size
        self.actor = create_squashed_normal_policy(
            self.task.observation_shape,
            self.task.action_size,
            check_encoder(self.hparams.encoder_factory),
        )
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.learning_rate)
        
        # build bc policy entropy_temp model
        entropy_temp = torch.ones((1, 1), dtype=torch.float32) * self.hparams.initial_entropy_temp
        self.log_entropy_temp = torch.nn.parameter.Parameter(torch.log(entropy_temp))    
        self.log_entropy_temp_optim = torch.optim.Adam([self.log_entropy_temp,], lr=self.hparams.learning_rate)
        
    def get_model(self):
        return self.actor
    
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        
        _, log_prob = self.actor(batch.obs,with_log_prob=True)
        entropy_temp_loss = -(self.log_entropy_temp.exp() * (log_prob + self.target_entropy_temp).detach()).mean()
        self.log_entropy_temp_optim.zero_grad()
        entropy_temp_loss.backward()
        self.log_entropy_temp_optim.step()

        policy_log_prob = self.actor.act_log_prob(batch.obs, batch.act)
        actor_loss = (self.log_entropy_temp.exp() * log_prob - policy_log_prob).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        loss = {
                'target_entropy' : self.target_entropy_temp,
                'bc_actor_loss': actor_loss.item(),
                'bc_alpha': self.log_entropy_temp.exp().cpu().detach().numpy()[0][0],
                'bc_alpha_loss': entropy_temp_loss.item(),
                'bc_log_probs': torch.mean(log_prob).item(),
            }
        self.log("bc_log_probs", loss['bc_log_probs'], on_step=True, on_epoch=True, prog_bar=True)
        self.log("bc_actor_loss", loss['bc_actor_loss'], on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def val_step(self,):
        pass
