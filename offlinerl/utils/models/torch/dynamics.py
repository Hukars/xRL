# pylint: disable=protected-access

from typing import List, Optional, Tuple, cast, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import linalg as LA
from torch.nn.utils import spectral_norm
from .transformer import ClassificationTransformer
from .encoders import EncoderWithAction


def _compute_ensemble_variance(
    observations: torch.Tensor,
    rewards: torch.Tensor,
    variances: torch.Tensor,
    variance_type: str,
) -> torch.Tensor:
    if variance_type == "max":
        return variances.max(dim=1).values
    elif variance_type == "data":
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1) ** 2).sum(dim=1, keepdim=True)
    raise ValueError(f"invalid variance_type: {variance_type}")


def _apply_spectral_norm_recursively(model: nn.Module) -> None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for m in module:
                _apply_spectral_norm_recursively(m)
        else:
            if "weight" in module._parameters:
                spectral_norm(module)


def _gaussian_likelihood(
    x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor
) -> torch.Tensor:
    inv_std = torch.exp(-logstd)
    return (((mu - x) ** 2) * inv_std).mean(dim=1, keepdim=True)


def _l2_loss(
    model: nn.Module
) -> torch.Tensor:
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += (LA.norm(param, 2)**2)
    return l2_loss


class DynamicsModel(nn.Module):  # type: ignore
    """Dynamics model which can be made probabilistic or determinstic
    References:
        * `Janner et al., When to Trust Your Model: Model-Based Policy
          Optimization. <https://arxiv.org/abs/1906.08253>`_
        * `Chua et al., Deep Reinforcement Learning in a Handful of Trials
          using Probabilistic Dynamics Models.
          <https://arxiv.org/abs/1805.12114>`_
    """

    _encoder: EncoderWithAction
    _mu: nn.Linear
    _logstd: nn.Linear
    _max_logstd: nn.Parameter
    _min_logstd: nn.Parameter

    def __init__(self, encoder: EncoderWithAction, deterministic=False):
        super().__init__()
        # apply spectral normalization except logstd encoder.
        _apply_spectral_norm_recursively(cast(nn.Module, encoder))
        self._encoder = encoder
        self._deterministic=deterministic

        feature_size = encoder.get_feature_size()
        observation_size = encoder.observation_shape[0]
        out_size = observation_size + 1

        # TODO: handle image observation
        self._mu = spectral_norm(nn.Linear(feature_size, out_size))
        if not deterministic:
            self._logstd = nn.Linear(feature_size, out_size)

            # logstd bounds
            init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
            init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
            self._max_logstd = nn.Parameter(init_max)
            self._min_logstd = nn.Parameter(init_min)

    def compute_stats(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        h = self._encoder(x, action)

        mu = self._mu(h)
        if self._deterministic:
            return mu
        else:
            # log standard deviation with bounds
            logstd = self._logstd(h)
            logstd = self._max_logstd - F.softplus(self._max_logstd - logstd)
            logstd = self._min_logstd + F.softplus(logstd - self._min_logstd)

            return mu, logstd

    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._deterministic:
            return self.predict_without_variance(x, action)
        else:
            return self.predict_with_variance(x, action)[:2]

    def predict_without_variance(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._deterministic
        mu = self.compute_stats(x, action)
        # residual prediction
        next_x = x + mu[:, :-1]
        next_reward = mu[:, -1].view(-1, 1)
        return next_x, next_reward

    def predict_with_variance(
        self, x: torch.Tensor, action: torch.Tensor, eval: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert not self._deterministic
        mu, logstd = self.compute_stats(x, action)
        dist = Normal(mu, logstd.exp())
        pred = dist.rsample()
        # residual prediction
        next_x = x + pred[:, :-1]
        next_reward = pred[:, -1].view(-1, 1)
        if not eval:
            return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)
        else:
            return x + mu[:, :-1], mu[:, -1].view(-1, 1), dist.variance.sum(dim=1, keepdims=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        obs_tp1: torch.Tensor,
        eval: bool = False,
    ) -> torch.Tensor:
        loss_func = nn.MSELoss()
        if self._deterministic:
            mu = self.compute_stats(obs_t, act_t)
            # residual prediction
            mu_x = obs_t + mu[:, :-1]
            mu_reward = mu[:, -1].view(-1, 1)
            mse_loss = loss_func(mu_x, obs_tp1)
            mse_loss += loss_func(mu_reward, rew_tp1)
            l2_loss = _l2_loss(self._mu) + _l2_loss(self._encoder)
            loss = mse_loss + 0.001 * l2_loss * 0.0
            return loss
        else:
            mu, logstd = self.compute_stats(obs_t, act_t)
            # residual prediction
            mu_x = obs_t + mu[:, :-1]
            mu_reward = mu[:, -1].view(-1, 1)
            logstd_x = logstd[:, :-1]
            logstd_reward = logstd[:, -1].view(-1, 1)

            if eval:
                mse_loss = loss_func(mu_x, obs_tp1)
                mse_loss += loss_func(mu_reward, rew_tp1)
                return mse_loss
            else:
                # gaussian likelihood loss
                likelihood_loss = _gaussian_likelihood(obs_tp1, mu_x, logstd_x)
                likelihood_loss += _gaussian_likelihood(
                    rew_tp1, mu_reward, logstd_reward
                )

                # penalty to minimize standard deviation
                penalty = logstd.sum(dim=1, keepdim=True)

                # minimize logstd bounds
                bound_loss = self._max_logstd.sum() - self._min_logstd.sum()

                loss = likelihood_loss + penalty + 1e-2 * bound_loss

                return loss.view(-1, 1)


class EnsembleDynamicsModel(nn.Module):  # type: ignore
    _models: nn.ModuleList

    def __init__(self, 
                 models: List[DynamicsModel],
                 transformer_encoder: ClassificationTransformer,
                 observation_shape: int, 
                 action_size: int,
                 latent_size: int,
                 encoder_size: int,
                 encoder_num_layers=3,):
        super().__init__()
        self._models = nn.ModuleList(models)
        self.base_transformer_encoder = transformer_encoder
        self.z = None
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.latent_size = latent_size
        self.activation = torch.nn.Tanh()
        self.encoder_num_layers = encoder_num_layers
        self._encoder = torch.nn.Sequential()
        encoder_sizes = (observation_shape + action_size + encoder_size, ) + (latent_size, ) * encoder_num_layers
        self._mu = torch.nn.Linear(latent_size, len(self._models))

        for i in range(encoder_num_layers):
            self._encoder.add_module(f'encoder_layer{i+1}',
                            torch.nn.Linear(encoder_sizes[i], encoder_sizes[i+1]))
            self._encoder.add_module(f'activation{i+1}', self.activation)
    
    def infer_from_context(self, context):
        with torch.no_grad():
            self.z = self.base_transformer_encoder.get_encoder_vector(context)

    def base_ensemble_predict(self, obs, action):
        next_observations_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        with torch.no_grad():
            for model in self._models:
                obs, rew = model.predict_without_variance(obs, action)
                next_observations_list.append(obs.view(1, obs.shape[0], -1))
                rewards_list.append(rew.view(1, obs.shape[0], 1))
         # (ensemble, batch, -1) -> (batch, ensemble, -1)
        next_observations = torch.cat(next_observations_list, dim=0).transpose(0, 1)
        rewards = torch.cat(rewards_list, dim=0).transpose(0, 1)    

        return next_observations, rewards

    def _get_weighted_predict(self, obs, action):
        next_observations, rewards = self.base_ensemble_predict(obs, action)
        obs = torch.cat([obs, action, self.z.repeat(obs.shape[0], 1)], 1)
        h = self._encoder(obs)
        dynamic_weight = self._mu(h)
        # print(rewards.shape)
        # print(dynamic_weight.shape)
        dynamic_weight = torch.sigmoid(dynamic_weight)
        weighted_rewards = torch.sum(rewards * 
            dynamic_weight.view(dynamic_weight.shape[0], -1, 1), dim=1)
        weighted_rewards = weighted_rewards / dynamic_weight.view(dynamic_weight.shape[0], -1, 1).abs().sum(1)

        # random pick one obs
        model_indexes = np.random.randint(0, next_observations.shape[1], size=(obs.shape[0]))
        next_observations = next_observations[np.arange(obs.shape[0]), model_indexes]

        return next_observations, weighted_rewards

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._get_weighted_predict(x, action)
    
    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        target_rew: torch.Tensor,
    ) -> torch.Tensor:
        _, predict_rew = self._get_weighted_predict(obs_t, act_t)
        loss_func = torch.nn.MSELoss()
        return loss_func(predict_rew, target_rew)

    @property
    def models(self) -> nn.ModuleList:
        return self._models
