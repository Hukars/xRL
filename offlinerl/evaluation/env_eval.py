import ray
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict

from offlinerl.utils.dataset import to_torch, to_numpy


# Faster but need more computation resources
@ray.remote 
def test_one_trail(env, policy, device, max_steps=1000, using_last_transition=False):
    state, done = env.reset(), False
    if using_last_transition:
        last_transition = np.concatenate([state[np.newaxis], 
                                         np.zeros((1, env.action_space.sample().shape[0])), 
                                         np.array([[0.0]]),
                                         state[np.newaxis]], 1)
    rewards = 0
    lengths = 0
    success_flag = False
    while (not done) and (lengths < max_steps):
        state = state[np.newaxis]
        if using_last_transition:
            action = policy.best_action(to_torch(state, dtype=torch.float32, device=device),
                                        to_torch(last_transition, dtype=torch.float32, device=device)).reshape(-1)
        else:
            action = policy.best_action(to_torch(state, dtype=torch.float32, device=device)).reshape(-1)
        next_state, reward, done, info = env.step(to_numpy(action.cpu()))
        if using_last_transition:
            last_transition = np.concatenate([state, 
                                            to_numpy(action.cpu())[np.newaxis], 
                                            np.array([[reward]]),
                                            next_state[np.newaxis]], 1)
        state = next_state
        lengths += 1
        rewards += reward
        success_flag = success_flag or info['success']

    return (rewards, lengths, success_flag)


def test_one_trail_without_ray(env, policy, device, max_steps=1000, using_last_transition=False):
    state, done = env.reset(), False
    if using_last_transition:
        last_transition = np.concatenate([state[np.newaxis], 
                                         np.zeros((1, env.action_space.sample().shape[0])), 
                                         np.array([[0.0]]),
                                         state[np.newaxis]], 1)
    rewards = 0
    lengths = 0
    success_flag = False
    while (not done) and (lengths < max_steps):
        state = state[np.newaxis]
        if using_last_transition:
            action = policy.best_action(to_torch(state, dtype=torch.float32, device=device),
                                        to_torch(last_transition, dtype=torch.float32, device=device)).reshape(-1)
        else:
            action = policy.best_action(to_torch(state, dtype=torch.float32, device=device)).reshape(-1)
        next_state, reward, done, info = env.step(to_numpy(action))
        if using_last_transition:
            last_transition = np.concatenate([state, 
                                            to_numpy(action.cpu())[np.newaxis], 
                                            np.array([[reward]]),
                                            next_state[np.newaxis]], 1)
        state = next_state
        lengths += 1
        rewards += reward
        success_flag = success_flag or info['success']

    return (rewards, lengths, success_flag)


def test_on_real_env(policy, env, device=torch.device('cpu'), max_env_steps=120, number_of_runs=5, using_ray=False, using_transition=False):
    if not ray.is_initialized() and using_ray:
        ray.init(ignore_reinit_error=True, _redis_max_memory=100000000)
    rewards = []
    episode_lengths = []
    env = deepcopy(env)
    policy = deepcopy(policy)
    if type(env) == tuple:
        if using_ray:
            results = ray.get([test_one_trail.remote(e, policy, device, max_env_steps, using_transition) for _ in range(number_of_runs) for e in env])
        else:
            results = [test_one_trail_without_ray(e, policy, device, max_env_steps, using_transition) for _ in range(number_of_runs) for e in env]
    else:
        if using_ray:
            results = ray.get([test_one_trail.remote(env, policy, device, max_env_steps, using_transition) for _ in range(number_of_runs)])
        else:
            results = [test_one_trail_without_ray(env, policy, device, max_env_steps, using_transition) for _ in range(number_of_runs)]

    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]
    success_flags = [result[2] for result in results] 
    rew_mean = np.mean(rewards)
    max_rew = np.max(rewards)
    min_rew = np.min(rewards)
    len_mean = np.mean(episode_lengths)
    success_rate = np.mean(success_flags)
    print(f'rewards:{rewards}')
    #print(f'success_rate:{success_rate}')

    res = OrderedDict()
    res["Reward_Mean_Env"] = rew_mean
    res["Length_Mean_Env"] = len_mean
    res['Success_Rate'] = success_rate
    res['Max_Reward'] = max_rew
    res['Min_Reward'] = min_rew
    
    if hasattr(env, "get_normalized_score"): 
        res["Score"] = env.get_normalized_score(rew_mean) * 100.0
    
    return res
