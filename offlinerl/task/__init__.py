import os
from tqdm import tqdm
import torch
import numpy as np
from loguru import logger
from dotted_dict import DottedDict
from offlinerl.utils.env_utils import get_env_shape
    

def sample_offline_datasets(env, policy_list, per_policy_episode_num, per_episode_max_length, save_path, save_flag='expert', device=None):
    """Use policy in policy list to sample per_policy_episode_num episodes in the given env

    """
    obs_list, next_obs_list, act_list, rew_list, done_list = [], [], [], [], []
    ep_num, success_num = 0, 0
    state, ep_rew, ep_len, success_flag = env.reset(), 0, 0, False
    for policy in policy_list:
        for e in tqdm(range(per_policy_episode_num)):
            last_ep_rew = ep_rew
            for t in range(per_episode_max_length+1):
                action = policy(torch.tensor(state, dtype=torch.float32) if device is None else
                    torch.tensor(state, dtype=torch.float32).to(device)).detach().cpu().numpy()
                next_state, reward, done, info = env.step(action)
                success_flag = success_flag or info['success']
                if done or ((t//per_episode_max_length)==1):
                    ep_num += 1 
                    success_num += int(success_flag)
                    print(f'success_num:{success_num}')
                    print(f'episode {e+1} reward: {ep_rew - last_ep_rew}')
                    state, success_flag = env.reset(), False
                else:
                    obs_list.append(state)
                    next_obs_list.append(next_state)
                    act_list.append(action)
                    rew_list.append(reward)
                    done_list.append(float(done))
                    state = next_state
                    ep_rew += reward
                    ep_len += 1
    print(f'mean_episode_reward:{ep_rew/ep_num:.2f}')
    print(f'success_rate: {success_num/ep_num:.2f}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.savez(os.path.join(save_path, f'{save_flag}.npz'), 
        obs=np.array(obs_list),
        obs_next=np.array(next_obs_list),
        act=np.array(act_list),
        rew=np.expand_dims(np.array(rew_list), 1),
        done=np.expand_dims(np.array(done_list), 1),)


def load_ml10_tasks(num=1):
    from .metaworld_ml10 import load_ml10_train_envs, load_ml10_test_envs
    train_envs, train_task_names = load_ml10_train_envs(num)
    test_envs, test_task_names = load_ml10_test_envs(num)
    if num == 1:
        return [DottedDict(check_task({'env': env, 'task_name': name}, False)) for env, name in zip(train_envs, train_task_names)], \
            [DottedDict(check_task({'env': env, 'task_name': name}, False)) for env, name in zip(test_envs, test_task_names)]
    elif num > 1:
        raise NotImplementedError('Do not use this branch!')
        # return [DottedDict(check_task({'envs': envs, 'task_name': names[0]}, False)) for envs, names in zip(zip(*train_envs), zip(*train_task_names))], \
        #     [DottedDict(check_task({'envs': envs, 'task_name': names[0]}, False)) for envs, names in zip(zip(*test_envs), zip(*test_task_names))]
    else:
        raise ValueError('num must be larger than 1')


def load_ml1_tasks(task_name):
    from .metaworld_ml1 import load_ml1_train_envs, load_ml1_test_envs
    train_envs = load_ml1_train_envs(task_name)
    test_envs = load_ml1_test_envs(task_name)
    return [DottedDict(check_task({'env': env, 'task_name': task_name}, False)) for env in train_envs], \
            [DottedDict(check_task({'env': env, 'task_name': task_name}, False)) for env in test_envs]

def load_d4rl_task(task_name):
    from .d4rl import load_d4rl_data_and_env
    task= load_d4rl_data_and_env(task_name)
    task['task_name'] = task_name
    task = check_task(task)
    
    return DottedDict(task)

def check_task(task, offline_benchmark=True):
    if "env" in task.keys():
        task["observation_shape"], task["action_size"] = get_env_shape(task["env"])

    if "envs" in task.keys():
        task["observation_shape"], task["action_size"] = get_env_shape(task["envs"][0])

    if "train_buffer" not in task.keys() and offline_benchmark:
        raise ValueError("The task should include 'train_buffer'!")
        
    if not ("observation_shape" in task.keys() and "action_size" in task.keys()):
        raise ValueError("The task should include 'env' or 'obs_shape and act_size'!")
    
    if (not ("observation_space" in task.keys() and "action_space" in task.keys())) and offline_benchmark:
        task["observation_space"] = [np.min(task["train_buffer"]["obs"],0),np.max(task["train_buffer"]["obs"],0)]
        task["action_space"] = [np.min(task["train_buffer"]["act"],0),np.max(task["train_buffer"]["act"],0)]
    
    return task