import copy
import torch
from copy import deepcopy
import offlinerl
import pytorch_lightning as pl
from offlinerl.algos import BaseOfflineAlgo
from offlinerl.utils.models.builders import create_squashed_normal_policy, create_continuous_q_function
from offlinerl.utils.models.check import check_encoder, check_q_func
from offlinerl.utils.torch_utils import soft_sync, nth_derivative
from offlinerl.utils.data import to_torch


class FBRC(BaseOfflineAlgo):
    def pretrain(self):
        kwargs = deepcopy(self.trainer.init_kwargs)
        kwargs['logger'] = pl.loggers.TensorBoardLogger(save_dir=kwargs['logger'].save_dir, name=f"log/bcp/{self.task.task_name}")
        trainer = offlinerl.trainer.BaseTrainer(self.task, 'bcp', **kwargs)
        pretrain_algo = trainer.train()
        return pretrain_algo.get_model().to(self.device)
    
    def build(self):
        # build bc policy model
        self._target_entropy_temp = -self.task.action_size
        
        # build policy model
        self._actor = create_squashed_normal_policy(
            self.task.observation_shape,
            self.task.action_size,
            check_encoder(self.hparams.actor_encoder_factory),
        )
        
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=self.hparams.actor_learning_rate) 
        
        # build policy entropy_temp model
        entropy_temp = torch.ones((1, 1), dtype=torch.float32) * self.hparams.initial_entropy_temp
        self._log_entropy_temp = torch.nn.parameter.Parameter(torch.log(entropy_temp))
        self._entropy_temp_optim = torch.optim.Adam([self._log_entropy_temp,], lr=self.hparams.entropy_temp_learning_rate)
        
        # build critic model
        self._critic = create_continuous_q_function(
            self.task.observation_shape,
            self.task.action_size,
            check_encoder(self.hparams.critic_encoder_factory),
            check_q_func(self.hparams.q_func_factory),
            n_ensembles=self.hparams.n_critics,
        )
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=self.hparams.critic_learning_rate)
        
        self._critic_target = copy.deepcopy(self._critic)

        self.bc_actor = copy.deepcopy(self._actor) 
    
    def setup(self, stage):
        self.bc_actor = self.pretrain()
    
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        
        batch.rew = batch.rew + self.hparams.reward_bonus
        critic_loss = self._update_critic(batch.obs, batch.act, batch.rew, batch.done, batch.obs_next)
        
        entropy_temp = self._update_entropy_temp(batch.obs)
        actor_loss = self._update_actor(batch.obs)
        
        soft_sync(self._critic_target, self._critic, self.hparams.tau)
        res = {
            'critic_loss' : critic_loss['critic_loss'].item(),
            'actor_loss': actor_loss['actor_loss'].item(),
            'entropy_loss': entropy_temp['entropy_loss'].item(),
            'entropy': entropy_temp['entropy'].item()
        }
        self.log("critic_loss", critic_loss['critic_loss'].item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("actor_loss", actor_loss['actor_loss'].item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("entropy_loss", entropy_temp['entropy_loss'].item(), on_step=True, on_epoch=False, prog_bar=True)
        self.log("entropy", entropy_temp['entropy'].item(), on_step=True, on_epoch=False, prog_bar=True)
        return res
    
    def val_step(self, batch):
        pass

    def _update_critic(self, obs, act, rew, done, obs_next):
        
        act_next_sample, _ = self._actor.sample_with_log_prob(obs_next)
        act_sample, _ = self._actor.sample_with_log_prob(obs)
        
        next_target_q1, next_target_q2 = self._dist_critic(
            obs_next, act_next_sample, target=True)
        
        target_q = rew + self.hparams.discount * (1-done) * torch.min(
            next_target_q1, next_target_q2)
        
        q1, q2 = self._dist_critic(obs, act, stop_gradient=True)
        
        q1_fn, q2_fn = self._critic.q_funcs
        
        q1_grads = nth_derivative(q1_fn(obs,act_sample).sum(), act_sample, 1) 
        q2_grads = nth_derivative(q2_fn(obs,act_sample).sum(), act_sample, 1)

        q1_grad_norm = torch.sum(torch.square(q1_grads), axis=-1)
        q2_grad_norm = torch.sum(torch.square(q2_grads), axis=-1)
        q_reg = torch.mean(q1_grad_norm + q2_grad_norm)

        loss = torch.nn.functional.mse_loss(target_q, q1) + torch.nn.functional.mse_loss(target_q, q2) + self.hparams.f_reg * q_reg
        
        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step() 
    
        return {
            'q1': torch.mean(q1).cpu().detach().numpy(),
            'q2': torch.mean(q2).cpu().detach().numpy(),
            'critic_loss': loss.cpu().detach().numpy(),
            'q1_grad': torch.mean(q1_grad_norm).cpu().detach().numpy(),
            'q2_grad': torch.mean(q2_grad_norm).cpu().detach().numpy(),
        }
    
    def _dist_critic(self, 
                    obs,
                    act,
                    target = False,
                    stop_gradient = False):
        if target:
            critic_model = self._critic_target
        else:
            critic_model = self._critic
            
        q1_fn, q2_fn = critic_model.q_funcs
        
        q1, q2 = q1_fn(obs, act), q2_fn(obs,act)
        log_probs = self.bc_actor.act_log_prob(obs, act)
        
        if stop_gradient:
            log_probs = log_probs.detach()
            
        return q1 + log_probs, q2 + log_probs
    
    def _update_entropy_temp(self, obs):
        with torch.no_grad():
            _, log_prob = self._actor.sample_with_log_prob(obs)
        loss = -(self._log_entropy_temp.exp() * (log_prob + self._target_entropy_temp)).mean()
        self._entropy_temp_optim.zero_grad()
        loss.backward()
        self._entropy_temp_optim.step()

        # current temperature value
        cur_entropy_temp = self._log_entropy_temp.exp().cpu().detach().numpy()[0][0]

        return {"entropy_loss" : loss.cpu().detach().numpy(), 
                "entropy" : cur_entropy_temp}
    
    def _update_actor(self,obs):
        self._critic.eval()
        self._actor_optim.zero_grad()
        
        act_sample, log_prob = self._actor.sample_with_log_prob(obs)
        q1, q2 = self._dist_critic(obs, act_sample)
        q = torch.min(q1, q2)
        entropy = self._log_entropy_temp.exp() * log_prob
        loss = (entropy - q).mean()       

        loss.backward() 
        self._actor_optim.step()
        
        return {"actor_loss" : loss.cpu().detach().numpy()}

    def get_model(self):
        return self._actor
