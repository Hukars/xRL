import metaworld


SEED = 0
def load_ml1_train_envs(choose_task, task_num=12):
    ml1 = metaworld.ML1(choose_task, seed=SEED)
    training_envs = []
    for i in range(task_num):
        env = ml1.train_classes[choose_task]()   
        task = ml1.train_tasks[i]
        env.set_task(task)
        training_envs.append(env)
    return training_envs


def load_ml1_test_envs(choose_task, task_num=8):
    ml1 = metaworld.ML1(choose_task, seed=SEED)
    testing_envs = []
    for i in range(task_num):
        env = ml1.test_classes[choose_task]()   
        task = ml1.test_tasks[i]
        env.set_task(task) 
        testing_envs.append(env)
    return testing_envs
