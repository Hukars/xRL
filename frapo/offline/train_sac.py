import os
import offlinerl
import pytorch_lightning as pl
from offlinerl.utils.torch_utils import select_free_cuda
from frapo.configs.global_parameters import BASE_DIR, SAC_POLICY_DIR


def sac_exp():
    """Online learning in given environments using SAC
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--is_test_envs', type=int, default=1)
    parser.add_argument('--task_index', type=int, default=0)
    parser.add_argument('--total_train_steps', type=int, default=2000000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--benchmark_type', type=str, default='ml1')
    parser.add_argument('--task_name', type=str, default='reach-v2')
    parser.add_argument('--pretrain_path', type=str, default=None)
     
    args = parser.parse_args()
    if args.benchmark_type == 'ml1':
        tasks = offlinerl.task.load_ml1_tasks(args.task_name)[int(args.is_test_envs)]
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    assert args.task_index <= (len(tasks)-1)
    args.gpu = select_free_cuda()
    task = tasks[args.task_index]
    task.flag = f'{args.benchmark_type}-{args.is_test_envs}-{args.task_index}'
    checkpoint_callback= pl.callbacks.ModelCheckpoint(dirpath=os.path.join(SAC_POLICY_DIR, f'{task.task_name}/{task.flag}'), filename='{step}-{val_mean_reward:.2f}',
                                                      every_n_train_steps=args.save_interval, save_top_k=-1, save_weights_only=True)
    log = pl.loggers.TensorBoardLogger(save_dir=BASE_DIR, name=f"log/sac/{task['task_name']}/{task['flag']}/")
    trainer = offlinerl.trainer.BaseTrainer(tasks[args.task_index], 
                                         'sac',
                                         seed=0,
                                         gpus=[args.gpu], 
                                         callbacks=[checkpoint_callback], 
                                         logger=log,
                                         limit_train_batches=args.total_train_steps)    
    trainer.train(pretrain_model_path=args.pretrain_path)


if __name__ =='__main__':
    sac_exp()