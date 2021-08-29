from typing import Tuple
import random
import numpy as np
from copy import deepcopy
from tianshou.data import Batch
from tianshou.data import to_torch, to_torch_as, to_numpy
from torch.utils.data import Dataset


def data_preprocessing_from_transition_buffer(buffer, 
                                              episode_length=120, 
                                              using_transformer=False, 
                                              segment_length=None,
                                              one_context=False,):
    """
        param buffer : the (obs, action, rew, next_obs, done) buffer
        param episode_length: the length of all the episodes(same)
        param using_transformer: whether to generate data to train a transformer
        param segment_length: the length of the segment(the number of transitions to form a segment)
        param one_context: whether to pick only one context to the offline data buffer
    """
    obs = buffer['obs']
    action = buffer['act']
    rew = buffer['rew']
    # Get (last_obs, last_action, obs)
    index = np.arange(0, obs.shape[0]) - 1
    start_index = np.arange(0, obs.shape[0], episode_length)
    index[start_index] = index[start_index] + 1
    last_obs = obs[index]
    last_action = action[index]
    last_reward = rew[index]
    last_action[start_index] = last_action[start_index] * 0.0
    last_reward[start_index] = last_reward[start_index] * 0.0
    buffer['last_transition'] = np.concatenate([last_obs, last_action, last_reward, obs], 1)
    if using_transformer:
        assert segment_length is not None
        assert obs.shape[0] % segment_length == 0
        buffer['context'] = buffer['last_transition'].reshape(obs.shape[0] // segment_length, segment_length, -1)
        if one_context:
            buffer['context'] = buffer['context'][[0]]
    return buffer


class SampleBatch(Batch):
    def sample(self, batch_size):
        length = len(self)
        assert 1 <= batch_size
        
        indices = np.random.randint(0, length, batch_size)
        
        return self[indices]


class RandomDataset(Dataset):
    def __init__(self, buffer: Batch):
        self.buffer = buffer
        
    def __len__(self):
        return len(self.buffer) * 1000
        
    def __getitem__(self, index):
        return dict(self.buffer[index])
    

class BatchDataset(Dataset):
    def __init__(self, buffer: Batch):
        self.buffer = buffer
        
    def __len__(self):
        return len(self.buffer)
        
    def __getitem__(self, index):
        return dict(self.buffer[index])
    
class TrajectoryDataset(Dataset):
    def __init__(self, buffer: Batch):
        self.buffer = buffer
        self.trajectory_end_indices = np.argwhere(self.buffer.reshape(-1) > 0.5).reshape(-1).astype("int32")
        self.trajectory_start_indices = np.delete(np.insert(self.trajectory_end_indices+1,0,0),-1).astype("int32")    
        
    def __len__(self):
        return int(np.sum(self.buffer.done))
        
    def __getitem__(self, index):
        return dict(self.buffer[self.trajectory_start_indices[index]:self.trajectory_end_indices[index]])


class InitDataset(Dataset):
    def __init__(self, buffer: Batch):
        self.buffer = buffer
        self.trajectory_end_indices = np.argwhere(self.buffer.reshape(-1) > 0.5).reshape(-1).astype("int32")
        self.trajectory_start_indices = np.delete(np.insert(self.trajectory_end_indices+1,0,0),-1).astype("int32")   
        
    def __len__(self):
        return int(np.sum(self.buffer.done))
        
    def __getitem__(self, index):
        return dict(self.buffer[self.trajectory_start_indices[index]])


class ReplayBufferDataset:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0
    
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size
    
    def get_length(self):
        return len(self.storge)

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)
    
    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    
    
def make_buffer_dataset(mode, *args, **kwargs):
    if mode == "random" or mode.lower() =="r":
        return RandomDataset(*args, **kwargs)
    elif mode == "batch" or mode.lower() == "b":
        return BatchDataset(*args, **kwargs)
    elif mode == "trajectory" or mode.lower() == "t":
        return TrajectoryDataset(*args, **kwargs)
    elif mode == "init" or mode.lower() == "i":
        return InitDataset(*args, **kwargs)
    elif mode == "replay" or mode.lower() == 're':
        return ReplayBufferDataset(*args, **kwargs)
    else:
        raise NotImplementedError("No specific class")