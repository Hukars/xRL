import copy
import math
import torch

from offlinerl.algos import BaseOfflineAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.torch_utils import soft_sync
from offlinerl.utils.models.torch.parameters import Parameter
from offlinerl.utils.models.check import check_encoder, check_q_func
from offlinerl.utils.models.builders import create_squashed_normal_policy, create_continuous_q_function


class CQL(BaseOfflineAlgo):
    
    def build(self) -> None:
        self._build_entropy_temp()
        self._build_actor()
        self._bulild_critic()
        self._build_alpha()
        
        self._critic_target = copy.deepcopy(self._critic)
        self._actor_target = copy.deepcopy(self._actor)
        
    def _build_alpha(self):
        alhpa = torch.ones((1, 1), dtype=torch.float32) * self.hparams.initial_alpha
        self._log_alpha = Parameter(torch.log(alhpa))           #?????
        self._log_alpha_optim = torch.optim.Adam(self._log_alpha.parameters(), lr=self.hparams.alpha_learning_rate)
        
    def _build_entropy_temp(self,):
        if self.hparams.auto_tune_entropy_temp:
            alhpa = torch.ones((1, 1), dtype=torch.float32) * self.hparams.initial_entropy_temp
            self._log_entropy_temp = Parameter(torch.log(alhpa))           #?????
            self._entropy_temp_optim = torch.optim.Adam(self._log_entropy_temp.parameters(), lr=self.hparams.entropy_temp_learning_rate)
        
    def _build_actor(self):
        actor_encoder_factory = check_encoder(self.hparams.actor_encoder_factory)
        
        self._actor = create_squashed_normal_policy(
            self.task.observation_shape,
            self.task.action_size,
            actor_encoder_factory,
        )
        
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=self.hparams.actor_learning_rate)
        
    def _bulild_critic(self):
        critic_encoder_factory = check_encoder(self.hparams.critic_encoder_factory)
        q_func_factory = check_q_func(self.hparams.q_func_factory)
        
        self._critic = create_continuous_q_function(
            self.task.observation_shape,
            self.task.action_size,
            critic_encoder_factory,
            q_func_factory,
            n_ensembles=self.hparams.n_critics,
        )
        
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=self.hparams.critic_learning_rate)
        
    def get_model(self):
        return self._actor
        
    def train_step(self, batch):
        batch = to_torch(batch, torch.float, device=self.device)
        critic_loss = self._update_critic(batch.obs, batch.act, batch.rew, batch.done, batch.obs_next)
        
        if self.global_step % self.hparams.update_actor_interval == 0:
            actor_loss = self._update_actor(batch.obs)
        
            if self.hparams.auto_tune_entropy_temp:
                entropy_temp_loss, entropy_temp = self._update_entropy_temp(batch.obs)
                
        soft_sync(self._critic_target, self._critic, self.hparams.tau)
        soft_sync(self._actor_target, self._actor, self.hparams.tau)
                
        return [critic_loss, actor_loss, entropy_temp_loss, entropy_temp]
    
    def val_step(self, batch):
        pass
        
    def _update_actor(self,obs):
        self._critic.eval()
        self._actor_optim.zero_grad()
        
        act_sample, log_prob = self._actor.sample_with_log_prob(obs)
        entropy = self._log_entropy_temp().exp() * log_prob
        q = self._critic(obs, act_sample, self.hparams.target_reduction_type)
        loss = (entropy - q).mean()       

        loss.backward() 
        self._actor_optim.step()
        
        return loss.cpu().detach().numpy()
    
    def _update_critic(self, obs, act, rew, done, obs_next):
        self._critic_optim.zero_grad()
        
        q_next = self._compute_target_q(obs_next)
        
        loss = self._critic.compute_error(obs, 
                                          act, 
                                          rew, 
                                          q_next, 
                                          done, 
                                          1, 
                                          use_independent_target=self.hparams.target_reduction_type == "none", 
                                          masks = None)
        
        conservative_loss = self._compute_conservative_loss(obs, act)
        
        loss += conservative_loss
        loss.backward()
        self._critic_optim.step() 
        
        return loss.cpu().detach().numpy()
    
    def _compute_conservative_loss(self, obs, act):
        with torch.no_grad():
            policy_actions, n_log_probs = self._actor.sample_n_with_log_prob(
                obs, self.hparams.n_action_samples
            )
            
        # policy action for t
        repeated_obs = obs.expand(self.hparams.n_action_samples, *obs.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs.shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.task.action_size,)

        # estimate action-values for policy actions
        policy_values = self._critic(flat_obs, flat_policy_acts, "none")
        policy_values = policy_values.view(
            self.hparams.n_critics, obs.shape[0], self.hparams.n_action_samples
        )
        log_probs = n_log_probs.view(1, -1, self.hparams.n_action_samples)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        random_actions = torch.zeros_like(flat_policy_acts).uniform_(-1.0, 1.0)
        random_values = self._critic(flat_obs, random_actions, "none")
        random_values = random_values.view(
            self.hparams.n_critics, obs.shape[0], self.hparams.n_action_samples
        )
        
        random_log_probs = math.log(0.5 ** self.task.action_size)
        #random_log_probs = torch.log(torch.ones([self._action_size],dtype=torch.float32) * 0.5)

        # compute logsumexp
        # (n critics, batch, 2 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values - log_probs, random_values - random_log_probs], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._critic(obs, act, "none")

        element_wise_loss = logsumexp - data_values - self.hparams.alpha_threshold

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)

        return (clipped_alpha[0][0] * element_wise_loss).sum(dim=0).mean()
    
    
    def _update_entropy_temp(self, obs):
        self._entropy_temp_optim.zero_grad()
        with torch.no_grad():
            _, log_prob = self._actor.sample_with_log_prob(obs)
            target_entropy_temp = log_prob - self.task.action_size

        loss = -(self._log_entropy_temp().exp() * target_entropy_temp).mean()

        loss.backward()
        self._entropy_temp_optim.step()

        # current temperature value
        cur_entropy_temp = self._log_entropy_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_entropy_temp
    
    def _update_alpha(self):
        self._critic.eval()
        
        self._log_alpha_optim.zero_grad()
        loss = -self._compute_conservative_loss(obs, act)
        loss.backward()
        
        self._log_alpha_optim.step()
        
        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]
        
        return loss.cpu().detach().numpy(), cur_alpha   
    
    def _compute_target_q(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, log_prob = self._actor.sample_with_log_prob(x)
            target_value = self._critic_target.compute_target(
                x, action, reduction=self.hparams.target_reduction_type
            )
            
            if self.hparams.soft_q_backup:
                entropy = self._log_entropy_temp().exp() * log_prob
                
                if self.hparams.target_reduction_type == "none":
                    target_value = target_value - entropy.view(1, -1, 1)
                else:
                    target_value = target_value - entropy
                
            return target_value
