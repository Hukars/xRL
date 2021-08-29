import offlinerl
import numpy as np
import os
from loguru import logger
import pytorch_lightning as pl
from frapo.configs.global_parameters import BASE_DIR, DATA_DIR, OFFLINE_POLICY_DIR
from offlinerl.utils.dataset import SampleBatch
from offlinerl.utils.torch_utils import select_free_cuda


def offline_exp():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--is_test_envs', type=int, default=0)
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--offline_algorithm', type=str, default='fbrc')
    parser.add_argument('--benchmark_type', type=str, default='ml1')
    parser.add_argument('--task_name', type=str, default='reach-v2')
    args = parser.parse_args()
    if args.benchmark_type == 'ml1':
        tasks = offlinerl.task.load_ml1_tasks(args.task_name)[int(args.is_test_envs)]
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    assert args.task_index <= (len(tasks)-1)
    task = tasks[args.task_index]
    args.gpu = select_free_cuda()
    offline_data_path = os.path.join(
        DATA_DIR,
        task.task_name,
        f'{args.benchmark_type}-{args.is_test_envs}',
        str(args.task_index),
    )
    offline_buffer = np.load(os.path.join(offline_data_path, 'expert.npz'))
    task.train_buffer = SampleBatch(
            obs=offline_buffer['obs'] ,
            obs_next=offline_buffer['obs_next'],
            act=offline_buffer['act'],
            rew=offline_buffer['rew'],
            done=offline_buffer['done'],
    )
    logger.info(f"{task.task_name}-{args.task_index}: {task.train_buffer['rew'].mean() * 120}")
    # assert False
    model_save_path = os.path.join(
        OFFLINE_POLICY_DIR,
        task.task_name,
        f'{args.benchmark_type}-{args.is_test_envs}',
        str(args.task_index),
    )
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    log = pl.loggers.TensorBoardLogger(save_dir=BASE_DIR, name=f"log/{args.offline_algorithm}/{args.task_name}/")
    checkpoint_callback= pl.callbacks.ModelCheckpoint(dirpath=model_save_path, filename='{epoch}-{val_mean_reward:.2f}', monitor='val_mean_reward', 
                                                      every_n_train_steps=8000, save_top_k=-1, save_weights_only=True)
    trainer = offlinerl.trainer.BaseTrainer(task, args.offline_algorithm, seed=0, gpus=[args.gpu], callbacks=[checkpoint_callback], logger=log)
    trainer.train()


if __name__ == "__main__":
    offline_exp()