import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from loguru import logger
from tianshou.data import Batch
from offlinerl.algos import BaseOnlineAlgo
from offlinerl.evaluation import test_on_real_env
from offlinerl.utils.torch_utils import soft_sync
from offlinerl.utils.models.torch.parameters import Parameter
from offlinerl.utils.models.check import check_encoder, check_q_func
from offlinerl.utils.models.builders import (create_squashed_normal_policy,
                                          create_continuous_q_function,)


class SAC(BaseOnlineAlgo):
    """Soft Actor-Critic implementation
       Reference:https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac
    """
    def sample_a_transition(self, env):
        obs = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
        if self.T > self.hparams.start_steps:
            action = self.actor(obs).cpu().numpy()
        else:
            action = env.action_space.sample()

        # Step the env
        next_obs, reward, d, _ = env.step(action)
        self.ep_ret += reward
        self.ep_len += 1

        self.done = False if self.ep_len==self.hparams.max_ep_length else d

        self.train_buffer.add(self.obs, action, reward, next_obs, float(self.done))
    
        self.obs = next_obs
        # End of trajectory handling
        if self.done or (self.ep_len == self.hparams.max_ep_length):
            logger.info(f'EpRet:{self.ep_ret}, EpLen:{self.ep_len}')
            self.obs, self.ep_ret, self.ep_len, self.done = env.reset(), 0, 0, None
    
    def sample_data_from_env(self):
        with torch.no_grad():
            env = self.task['env']
            if self.T < self.hparams.update_after:
                for _ in range(self.hparams.update_after):
                    self.sample_a_transition(env)
                self.T = self.hparams.update_after
            else:
                self.sample_a_transition(env)
                self.T += 1
    
    def _build_alpha(self):
        alpha = torch.ones((1, 1), device=self.device) * self.hparams.initial_alpha
        self.log_alpha = nn.Parameter(torch.log(alpha)) 
        self.log_alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.hparams.alpha_learning_rate)
        
    def _bulild_critic(self):
        critic_encoder_factory = check_encoder(self.hparams.critic_encoder_factory)
        q_func_factory = check_q_func(self.hparams.q_func_factory)
        
        self.critic = create_continuous_q_function(
            self.task.observation_shape,
            self.task.action_size,
            critic_encoder_factory,
            q_func_factory,
            n_ensembles=self.hparams.n_critics,
        )
        self.target_critic = deepcopy(self.critic)
        
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.hparams.critic_learning_rate)
        
    def _build_actor(self):
        actor_encoder_factory = check_encoder(self.hparams.actor_encoder_factory)
        
        self.actor = create_squashed_normal_policy(
            self.task.observation_shape,
            self.task.action_size,
            actor_encoder_factory,
        )
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.hparams.actor_learning_rate)
        
    def build(self) -> None:
        """Build the model used in the policy optimization stage
        """
        self._build_actor()
        self._bulild_critic()
        self._build_alpha()
        if "target_entropy" not in self.hparams.keys():
            self.hparams["target_entropy"] = -1 * self.task.env.action_space.shape[0]
        self.T = 0 # global time steps(sample steps)
        self.obs, self.ep_ret, self.ep_len, self.done = self.task.env.reset(), 0, 0, None
    
    def setup(self, stage):
        self.sample_data_from_env()

    def _sac_update(self, batch_data):
        obs = torch.tensor(batch_data[0], dtype=torch.float32, device=self.device)
        action = torch.tensor(batch_data[1], dtype=torch.float32, device=self.device)
        reward = torch.tensor(batch_data[2], dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.tensor(batch_data[3], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch_data[4], dtype=torch.float32, device=self.device).unsqueeze(-1)

        # update critic
        perdict_q_value = self.critic(obs, action, reduction="none")

        with torch.no_grad():
            next_action, log_prob = self.actor.sample_with_log_prob(next_obs)
            target_q_value = self.target_critic.compute_target(next_obs, next_action, reduction='min')
            alpha = torch.exp(self.log_alpha)
            y = reward + self.hparams.discount * (1 - done) * (target_q_value - alpha * log_prob)
        
        critic_loss = ((perdict_q_value - y.unsqueeze(0).repeat(2, 1, 1))**2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        if self.T % self.hparams.soft_update_frequency == 0:
            soft_sync(self.target_critic, self.critic, tau=self.hparams.soft_target_tau)
        
        if self.hparams.learnable_alpha:
            # update alpha
            alpha_loss = -torch.mean(self.log_alpha * (log_prob + self.hparams.target_entropy).detach())

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # update actor
        new_action, action_log_prob = self.actor.sample_with_log_prob(obs)
        q = self.critic(obs, new_action, reduction='min')
        actor_loss = - q.mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'log_alpha': self.log_alpha.data.item(),
            'entropy': -action_log_prob.mean().item()
        }
        
    def train_step(self, batch):
        batch = self.train_buffer.sample(self.hparams.batch_size)
        res = self._sac_update(batch)
        self.log("critic_loss", res['critic_loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("actor_loss", res['actor_loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("log_alpha", res['log_alpha'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("entropy", res['entropy'], on_step=True, on_epoch=False, prog_bar=True)
        return res
    
    def train_step_end(self, training_step_outputs):
        self.sample_data_from_env()
        if ((self.T-1) - self.hparams.update_after) % self.hparams.display_interval == 0:
            self.eval_policy()
    
    def eval_policy(self,):
        res = test_on_real_env(self.get_model(), self.task["env"], self.device, max_env_steps=self.hparams.max_ep_length if 'max_ep_length' in self.hparams.keys() else 1000,
            number_of_runs=2, using_ray=False)
        logger.info(f"result: {res} ")
        self.log("val_mean_reward", res['Reward_Mean_Env'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("Length_Mean_Env", res['Length_Mean_Env'], on_step=True, on_epoch=False, prog_bar=True)
        self.log("Success_Rate", res['Success_Rate'], on_step=True, on_epoch=False, prog_bar=True)
        return res

    def get_model(self):
        return self.actor
