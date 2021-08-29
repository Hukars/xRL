import metaworld
import random


SEED = 0
ml10 = metaworld.ML10(seed=SEED) # Construct the ML10 benchmark


def load_ml10_train_envs(num, random=False):
  if not random:
    assert num == 1
  training_envs = []
  training_task_names = []
  for _ in range(num):
    n_training_envs = []
    n_training_task_names =[]
    for name, env_cls in ml10.train_classes.items():
      env = env_cls()
      task_list = [task for task in ml10.train_tasks if task.env_name == name]
      if random:
        task = random.choice(task_list)
      else:
        task = task_list[0]
      env.set_task(task)
      n_training_envs.append(env)
      n_training_task_names.append(name)
    if num == 1:
      return n_training_envs, n_training_task_names
    else:
      training_envs.append(n_training_envs), training_task_names.append(n_training_task_names)
  return training_envs, training_task_names


def load_ml10_test_envs(num, random=False):
  if not random:
    assert num == 1
  testing_envs = []
  testing_task_names = []
  for _ in range(num):
    n_testing_envs = []
    n_testing_task_names =[]
    for name, env_cls in ml10.test_classes.items():
      env = env_cls()
      task_list = [task for task in ml10.test_tasks if task.env_name == name]
      if random:
        task = random.choice(task_list)
      else:
        task = task_list[0]
      env.set_task(task)
      n_testing_envs.append(env)
      n_testing_task_names.append(name)
    if num == 1:
        return n_testing_envs, n_testing_task_names
    else:
      testing_envs.append(n_testing_envs), testing_task_names.append(n_testing_task_names)
  return testing_envs, testing_task_names
