import torch
import numpy as np
from typing import List, cast
from collections import OrderedDict
from torch.distributions import Normal
from offlinerl.utils.torch_utils import soft_clamp
from .policies import Policy, squash_action
from .transformer import ClassificationTransformer


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

        self.register_parameter('saved_weight', torch.nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', torch.nn.Parameter(self.bias.detach().clone()))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes
        self.weight.data[indexes] = self.saved_weight.data[indexes]
        self.bias.data[indexes] = self.saved_bias.data[indexes]

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]


class EnsembleWeight(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, weight_dim, hidden_features, hidden_layers, ensemble_size=5):
        super().__init__()
        self.activation = torch.nn.Tanh()
        # self.activation = torch.nn.Identity()
        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        # self.output_layer = EnsembleLinear(hidden_features, 2*weight_dim, ensemble_size)
        # self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(weight_dim) * 1, requires_grad=True))
        # self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(weight_dim) * -5, requires_grad=True))
        self.output_layer = EnsembleLinear(hidden_features, weight_dim, ensemble_size)

    def forward(self, obs_action):
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
      
        return self.output_layer(output)
    
    def dist(self, obs_action, kappa):
        probs = torch.softmax(self.forward(obs_action) * kappa, dim=-1)
        return torch.distributions.Categorical(probs)
    
    def compute_loss(self, obs_action, target):
        pass
        # x = self.forward(obs_action)
        # x = x.permute((1, 2, 0))
        # loss_func = torch.nn.CrossEntropyLoss()
        # loss = loss_func(x, target.repeat(x.shape[-1], 1).transpose(0, 1)).mean()
        # return loss
    
    def weight_use(self, obs_action, ensemble_next_obs, ensemble_reward, kappa=2):
        dist = self.dist(obs_action, kappa)
        m = dist.probs.shape[0]
        weight = dist.probs
        weight = weight.view(m, obs_action.shape[0], -1, 1)
        next_obs, reward = torch.sum(ensemble_next_obs * weight, dim=2), \
            torch.sum(ensemble_reward * weight, dim=2)
        weight_indexes = np.random.randint(0, next_obs.shape[0], size=(obs_action.shape[0]))
        weight_next_obs, weight_reward = next_obs[weight_indexes, np.arange(obs_action.shape[0])],\
            reward[weight_indexes, np.arange(obs_action.shape[0])]
        return weight_next_obs, weight_reward


class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = Swish()

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)

        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def forward(self, obs_action):
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)


class EnsemblePolicy(torch.nn.Module):
    def __init__(self, 
                 base_policy_list: List[Policy], 
                 transformer_encoder: ClassificationTransformer,
                 observation_shape: int, 
                 action_size: int, 
                 latent_size: int, 
                 encoder_size: int,
                 min_logstd: float,
                 max_logstd: float,
                 without_latent_vector: bool,
                 encoder_num_layers=3,):
        super().__init__()
        self.base_policys = torch.nn.ModuleList(base_policy_list)
        self.base_transformer_encoder = transformer_encoder
        self.z = None
        self.observation_shape = observation_shape #todo: configurable observation
        self.env_action_size = action_size
        self.action_size = action_size * len(self.base_policys)
        self.encoder_size = encoder_size
        self.latent_size = latent_size
        self.activation = torch.nn.Tanh()
        self.encoder_num_layers = encoder_num_layers
        self._encoder = torch.nn.Sequential()
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._without_latent_vector = without_latent_vector

        encoder_sizes = (observation_shape + encoder_size, ) + (latent_size, ) * encoder_num_layers    
        self._mu = torch.nn.Linear(latent_size, self.action_size)
        self._logstd = torch.nn.Linear(latent_size, self.env_action_size)

        for i in range(encoder_num_layers):
            self._encoder.add_module(f'encoder_layer{i+1}',
                            torch.nn.Linear(encoder_sizes[i], encoder_sizes[i+1]))
            self._encoder.add_module(f'activation{i+1}', self.activation)
    
    def infer_from_context(self, context):
        # with torch.no_grad():
        self.z = self.base_transformer_encoder.get_encoder_vector(context)
        
    def _compute_logstd(self, h: torch.Tensor, new_weight=None):
        if new_weight is None:
            logstd = cast(torch.nn.Linear, self._logstd)(h)
        else:
            logstd = torch.nn.functional.linear(
                h,
                weight=new_weight['_logstd.weight'],
                bias=new_weight['_logstd.bias']
            )
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd
    
    def dist(self, mu, h: torch.Tensor, new_weight=None) -> Normal:
        clipped_logstd = self._compute_logstd(h, new_weight)
        return Normal(mu, clipped_logstd.exp())
    
    def base_ensemble_action(self, obs):
        action_list = []
        with torch.no_grad():
            for policy in self.base_policys:
                action_list.append(policy.best_action(obs).view(1, obs.shape[0], -1))
        # (ensemble, batch, -1) -> (batch, ensemble, -1)
            ensemble_action = torch.cat(action_list, 0).transpose(0, 1)
        return ensemble_action
    
    def forward(self, obs, obs_transition=None, new_weight=None, deterministic: bool = False,
                with_log_prob: bool = False,):
        mu_action, h = self._get_weighted_action(obs, obs_transition, new_weight)
        dist = self.dist(mu_action, h, new_weight)
        if deterministic:
            action = dist.loc
        else:
            action = dist.rsample()

        if with_log_prob:
            return squash_action(dist, action)  
            
        squashed_action = torch.tanh(action)
        return squashed_action
    
    def _get_weighted_action(self, obs, obs_transition=None, new_weight=None):
        ensemble_action = self.base_ensemble_action(obs)
        if obs_transition is not None:
            obs = obs_transition
        if not self._without_latent_vector:
            obs = torch.cat([obs, self.z.repeat(obs.shape[0], 1).to(obs.device)], 1)
        # else:
        #     obs = torch.cat([obs, torch.zeros(obs.shape[0], self.encoder_size).to(obs.device)], 1)
        if new_weight is None:
            h = self._encoder(obs)
            dynamic_weight = self._mu(h)
        else:
            h = obs
            for i in range(self.encoder_num_layers):
                h = torch.nn.functional.linear(
                    h,
                    weight=new_weight[f'encoder_layer{i+1}.weight'],
                    bias=new_weight[f'encoder_layer{i+1}.bias']
                )
                h = self.activation(h)
            dynamic_weight = torch.nn.functional.linear(
                h,
                weight=new_weight['_mu.weight'],
                bias=new_weight['_mu.bias']
            )
        dynamic_weight = torch.tanh(dynamic_weight)
        weighted_action = torch.sum(ensemble_action * 
            dynamic_weight.view(dynamic_weight.shape[0], -1, self.env_action_size), dim=1)
        weighted_action = weighted_action / dynamic_weight.view(dynamic_weight.shape[0], -1, self.env_action_size).abs().sum(1)

        return weighted_action, h

    def best_action(self, obs, obs_transition=None):
        return self._get_weighted_action(obs, obs_transition)[0]
    
    def act_log_prob(self, obs, target_action, obs_transition=None, new_weight=None, with_entropy=False):
        target_action = torch.clip(target_action, -1 + 1e-6, 1 - 1e-6)
        mu_action, h = self._get_weighted_action(obs, obs_transition, new_weight=new_weight)
        dist = self.dist(mu_action, h, new_weight)
        sample_act = torch.tanh(dist.rsample())
        sample_act = torch.clip(sample_act, -1 + 1e-6, 1 - 1e-6)
        
        if with_entropy:
            return squash_action(dist, torch.atanh(target_action))[1], - squash_action(dist, torch.atanh(sample_act))[1]
        else:
            return squash_action(dist, torch.atanh(target_action))[1]
    
    def act_mse(self, obs, target_action, obs_transition=None, new_weight=None):
        target_action = torch.clip(target_action, -1 + 1e-6, 1 - 1e-6)
        mu_action, _ = self._get_weighted_action(obs, obs_transition, new_weight)
        loss_func = torch.nn.MSELoss()
        return loss_func(mu_action, target_action)

    def update_params(self, loss, lr, deterministic_loss=False, params=None, first_order=True):
        if params is None:
            params = OrderedDict(self._encoder.named_parameters())

            # add mu
            mu_d = OrderedDict(self._mu.named_parameters())
            mu_d = OrderedDict([(f'_mu.{k}', v) for k, v in mu_d.items()])
            params.update(mu_d)
            # add std
            if not deterministic_loss:
                std_d = OrderedDict(self._logstd.named_parameters())
                std_d = OrderedDict([(f'_logstd.{k}', v) for k, v in std_d.items()])
                params.update(std_d)
           
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - lr * grad

        return updated_params

    @property
    def weightlength(self):
        return len(self.base_policys) * self.action_size
