import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.dataset import SampleBatch
from tianshou import data

ratios = [0.01, 0.01, 0.10, 0.88]

def load_d4rl_whole_range_data(task_prefix, data_size):
    task_sizes = [int(data_size*r) for r in ratios]
    online_data_buffer = SampleBatch()
    for i, task in enumerate([task_prefix+'-'+suffix+'-v0' for suffix in ['random', 'medium-replay', 'medium', 'expert']]):
        env = gym.make(task)
        dataset = d4rl.qlearning_dataset(env)

        total_buffer = SampleBatch(
            obs=dataset['observations'],
            obs_next=dataset['next_observations'],
            act=dataset['actions'],
            rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
            done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
        )

        online_data_buffer.cat_(total_buffer.sample(task_sizes[i]))
    return online_data_buffer
        

def load_d4rl_data_and_env(task):
    env = gym.make(task)
    dataset = d4rl.qlearning_dataset(env)

    train_buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )
    
    logger.info('train_buffer obs shape: {}', train_buffer.obs.shape)
    logger.info('train_buffer obs_next shape: {}', train_buffer.obs_next.shape)
    logger.info('train_buffer act shape: {}', train_buffer.act.shape)
    logger.info('train_buffer rew shape: {}', train_buffer.rew.shape)
    logger.info('train_buffer done shape: {}', train_buffer.done.shape)
    logger.info('Episode reward: {}', train_buffer.rew.sum() /np.sum(train_buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(train_buffer.done))
    
    task = {
        "train_buffer" : train_buffer,
        "env" : env,
    }
    
    return task