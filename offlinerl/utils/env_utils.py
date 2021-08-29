import gym
# import neorl


# def get_env(task):
#     try:
#         if task.startswith("HalfCheetah-v3"):
#             env = neorl.make("HalfCheetah-v3")
#         elif task.startswith("Hopper-v3"):
#             env = neorl.make("Hopper-v3")
#         elif task.startswith("Walker2d-v3"):   
#             env = neorl.make("Walker2d-v3")
#         elif task.startswith('d4rl'):
#             import d4rl
#             env = gym.make(task[5:])
#         else:
#             task_name = task.strip().split("-")[0]
#             env = neorl.make(task_name)
#     except:
#             raise NotImplementedError

#     return env

def get_env_shape(env):
    obs_shape = env.observation_space.shape
    act_size = env.action_space
        
    if hasattr(env.action_space, 'n'):
        act_size = env.action_space.n
    else:
        act_size = env.action_space.shape[0]
    
    return list(obs_shape), act_size

def get_env_action_range(task):
    env = get_env(task)
    act_max = float(env.action_space.high[0])
    act_min = float(env.action_space.low[0])
    
    return act_max, act_min  
    
def get_env_state_range(task):
    env = get_env(task)
    obs_max = float(env.observation_space.high[0])
    obs_min = float(env.observation_space.low[0])
    
    return obs_max, obs_min