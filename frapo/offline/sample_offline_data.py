import offlinerl
import os
import torch
from loguru import logger
from frapo.configs.global_parameters import DATA_DIR, SAC_POLICY_DIR


if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--is_test_envs', type=int, default=1)
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='expert')
    parser.add_argument('--sample_episode_num', type=int, default=10000)
    parser.add_argument('--sample_max_path_length', type=int, default=120)
    parser.add_argument('--benchmark_type', type=str, default='ml1')
    parser.add_argument('--task_name', type=str, default='reach-v2')
    args = parser.parse_args()
    if args.benchmark_type == 'ml1':
        tasks = offlinerl.task.load_ml1_tasks(args.task_name)[int(args.is_test_envs)]
    elif args.benchmark_type == 'ml10':
        tasks = offlinerl.task.load_ml10_tasks()[int(args.is_test_envs)]
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    assert args.task_index <= (len(tasks)-1)

    device = torch.device(f'cuda:{args.gpu}')

    trainer = offlinerl.trainer.BaseTrainer(tasks[args.task_index], 
                                         'sac', seed=0, need_log=False)
    model_path = os.path.join(
        SAC_POLICY_DIR, 
        f'{args.task_name}',
        f'{args.benchmark_type}-{args.is_test_envs}-{args.task_index}'
    )
    offline_data_save_path = os.path.join(
        DATA_DIR,
        trainer._task["task_name"],
        f'{args.benchmark_type}-{args.is_test_envs}',
        str(args.task_index)
    )
    logger.info(f'Here are the model to sample data: {model_path}')
    if not os.path.exists(model_path):
        raise FileNotFoundError('Please pretrain a policy model using ./train_sac.py')
    else:
        model_list = os.listdir(model_path)
        d = [(float((name.split('.')[0]).split('=')[-1]), name)for name in model_list]
        choose_model = sorted(d, key=lambda x:x[0])[-1][-1]
        model_path = os.path.join(model_path, choose_model)
        print(f'----------------model_path:{model_path}')
        trainer.on_before_train()
        trainer.algo.load_state_dict(torch.load(model_path)['state_dict'])
        policy = trainer.algo.get_model().to(device)
    offlinerl.task.sample_offline_datasets(env=tasks[min(args.task_index, len(tasks)-1)].env, 
                                        policy_list=[policy], 
                                        per_policy_episode_num=args.sample_episode_num,
                                        per_episode_max_length=args.sample_max_path_length,
                                        save_path=offline_data_save_path,
                                        save_flag=args.model_type,
                                        device=device)
 