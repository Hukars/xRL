import torch
import time
import os
from loguru import logger
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from configs.global_parameters import BASE_DIR, DATA_DIR, OFFLINE_POLICY_DIR
from offlinerl.task import load_ml1_tasks
from offlinerl.trainer import BaseTrainer
from offlinerl.utils.models.builders import (create_ensemble_policy_model, create_transformer_classification_model,
    create_lstm_classification_model, create_transformer_for_rl)
from offlinerl.evaluation import test_on_real_env
from offlinerl.utils.conf_utils import get_default_hparams, deep_update_dict
from offlinerl.utils.torch_utils import setup_seed, soft_sync, select_free_cuda
from offlinerl.utils.dataset import data_preprocessing_from_transition_buffer, SampleBatch

class WeightSearch:
    def __init__(self, base_policy_list, adapt_tasks, meta_test_tasks, hparams, device, log):
        self.adapt_tasks = adapt_tasks
        self.meta_test_tasks = meta_test_tasks
        self.device = device
        self.log = log
        self.hparams = hparams 
        self.n_sample_steps_total = 0

        if self.hparams.encoder_type == 'transformer':
            self.context_encoder = create_transformer_classification_model(
                self.adapt_tasks[0].observation_shape,
                self.adapt_tasks[0].action_size,
                self.hparams.encoder_size,
                self.hparams.test_buffer_size,
                len(self.hparams.adapt_task_indices),
                device,
            )
        elif self.hparams.encoder_type == 'lstm':
            self.context_encoder = create_lstm_classification_model(
                self.adapt_tasks[0].observation_shape,
                self.adapt_tasks[0].action_size,
                self.hparams.encoder_size,
                len(self.hparams.adapt_task_indices),
                device,
            )
        elif self.hparams.encoder_type == 'transformer_rl':
            self.context_encoder = create_transformer_for_rl(
                self.adapt_tasks[0].observation_shape, 
                self.adapt_tasks[0].action_size,
                self.hparams.encoder_size,
                self.hparams.test_buffer_size, 
                device
            ).to(device)
        else:
            raise NotImplementedError('No such model!')

        if self.hparams.pretrained_encoder_path is not None:
            print(self.hparams.pretrained_encoder_path)
            self.context_encoder.load_state_dict(torch.load(self.hparams.pretrained_encoder_path))
            self.context_encoder.eval()
        else:
            self.context_encoder.train()

        policy_encoder_size = self.hparams.encoder_size if self.hparams.policy_type == 'vector' else 0
        self.ensemble_policy = create_ensemble_policy_model(
                base_policy_list, self.context_encoder, self.adapt_tasks[0].observation_shape, 
                self.adapt_tasks[0].action_size, policy_encoder_size, self.hparams.policy_type).to(device)
        self.test_ensemble_policy = deepcopy(self.ensemble_policy)
        self.ensemble_weight_optim = torch.optim.Adam(self.ensemble_policy.parameters(), lr=hparams.update_lr)
        
    def policy_loss_func(self, batch, meta_test=False):
        batch.to_torch(dtype=torch.float32, device=self.device)
        if self.hparams.policy_type == 'transition':
            obs_transition = batch['last_transition']
        else:
            obs_transition = None
        obs = batch['obs']
        action = batch['act']
        if not meta_test:
            actor_loss = self.ensemble_policy.act_mse(obs, action, obs_transition)
        else:
            actor_loss = self.test_ensemble_policy.act_mse(obs, action, obs_transition)
        
        return actor_loss

    def meta_training(self, ):
        for t in tqdm(range(self.hparams.meta_train_gradient_steps)):
            # do meta testing in train tasks
            # if t % self.hparams.eval_algo_interval == 0:
            #     self.do_eval(t, eval_flag='Train')

            # do meta testing in test tasks
            if t % self.hparams.eval_algo_interval == 0:
               mean_test_tasks_reward = self.do_eval(t)

            # save model
            if (t % self.hparams.save_model_interval == 0) and (self.hparams.save_model):
                save_path = os.path.join(self.hparams.save_path, 'models')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.ensemble_policy.state_dict(), os.path.join(save_path, f'policy_steps_{t}_{mean_test_tasks_reward}.pt'))

            all_tasks_loss = 0.0
            choose_task_index = np.random.randint(0, len(self.adapt_tasks), self.hparams.meta_batch)
            for index in choose_task_index:
                task = self.adapt_tasks[index]
                batch = task.data_buffer.sample(self.hparams.batch_size+1)
                update_batch = batch[0:self.hparams.batch_size]
                # update z from random context
                context = torch.from_numpy(batch['context'][[-1]]).float().to(self.device)
                if self.hparams.pretrained_encoder_path is not None:
                    with torch.no_grad():
                        self.ensemble_policy.infer_from_context(context)
                elif self.hparams.policy_type == 'vector':
                    self.ensemble_policy.infer_from_context(context)
                else:
                    pass
                all_tasks_loss += self.policy_loss_func(update_batch)
                self.n_sample_steps_total += self.hparams.batch_size

            # optimize policy
            avg_task_loss = all_tasks_loss / self.hparams.meta_batch
            self.ensemble_weight_optim.zero_grad()
            avg_task_loss.backward()
            self.ensemble_weight_optim.step()

    def do_eval(self, s, eval_flag='Test'):
        """Do evaluation at the real environment
            param s: current training steps
            param eval_flag: 'Test' means test tasks; 'Train' means train tasks
        """
        mean_env_reward_list = []
        mean_success_rate_list = []
        max_reward_list = []
        min_reward_list = []
        logger.info(f"Meta Test Stage in {eval_flag} tasks: using sgd to minimize model bias")
        eval_tasks = self.meta_test_tasks if eval_flag == 'Test' else self.adapt_tasks
        for i, task in enumerate(eval_tasks):
            soft_sync(self.test_ensemble_policy, self.ensemble_policy, 1)
            batch = task.data_buffer.sample(1)
            self.test_ensemble_policy.eval()
            # update z from offline context
            if self.test_ensemble_policy.z is None:
                # In meta testing, our algorithm just need one context all the time
                context = torch.from_numpy(batch['context'][[-1]]).float().to(self.device)
                if self.hparams.policy_type == 'vector':
                    with torch.no_grad():
                        self.test_ensemble_policy.infer_from_context(context) 
            res = self.eval_ensemble_policy(s, i, task.env, eval_flag)
            self.test_ensemble_policy.z=None
            self.log.add_scalar(f'{eval_flag}_Mean_Env_Reward_{i+1}', res['Reward_Mean_Env'], s)
            mean_env_reward_list.append(res['Reward_Mean_Env'])
            mean_success_rate_list.append(res['Success_Rate'])
            max_reward_list.append(res['Max_Reward'])
            min_reward_list.append(res['Min_Reward'])
        self.log.add_scalar(f'{eval_flag}_Mean_Env_Reward', np.mean(mean_env_reward_list), s)
        self.log.add_scalar(f'{eval_flag}_Mean_Success_Rate', np.mean(mean_success_rate_list), s)
        self.log.add_scalar(f'{eval_flag}_Max_Reward', np.max(max_reward_list), s)
        self.log.add_scalar(f'{eval_flag}_Min_Reward', np.min(min_reward_list), s)
        return np.mean(mean_env_reward_list)
         
    def eval_ensemble_policy(self, iter, index, env, flag):
        res = test_on_real_env(self.test_ensemble_policy, env, self.device, using_ray=self.hparams.using_ray, using_transition=(self.hparams.policy_type=='transition'))
        logger.info(f"epoch：{iter}, {flag}_env_{index}, val_mean_reward: {res['Reward_Mean_Env']}, Length_Mean_Env: {res['Length_Mean_Env']},"
                    f"Success_Rate：{res['Success_Rate']}, Max_Reward：{res['Max_Reward']}， Min_Reward：{res['Min_Reward']}"
        )
        return res


def frapo_train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_path', type=str, default='frapo/configs/vector/trans_negt/reach.yaml')
    args = parser.parse_args()
    args.gpu = select_free_cuda()
    device = torch.device(f'cuda:{args.gpu}')
    setup_seed(args.seed)

    default_conf = get_default_hparams(OmegaConf.load(os.path.join(
                BASE_DIR,
                'frapo/configs/meta_learning/default.yaml'
            )
        )
    )
    conf = get_default_hparams(OmegaConf.load(os.path.join(
                BASE_DIR,
                args.config_path
            )
        )
    )
    conf = deep_update_dict(conf, default_conf)
    # 1. load the task
    if conf.benchmark_type == 'ml1':
        train_tasks, test_tasks = load_ml1_tasks(conf.task_name) 
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    
    #2. load base policy
    base_policy_list = []
    for index in conf.base_task_indices:
        trainer = BaseTrainer(train_tasks[index], conf.offline_algo, gpus=[args.gpu], seed=args.seed, need_log=False)
        trainer.on_before_train()
        model_path = os.path.join(
            OFFLINE_POLICY_DIR,
            conf.task_name if conf.benchmark_type == 'ml1' else train_tasks[index].task_name,
            f'{conf.benchmark_type}-0',
            str(index),
            'policy.ckpt'
        )
        trainer.algo.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        base_policy_list.append(trainer.algo.get_model().to(device))
    
    #3. load the data of the adapt tasks and test task
    adapt_train_tasks = [train_tasks[index] for index in conf.adapt_task_indices]
    meta_test_tasks = [test_tasks[index] for index in conf.test_task_indices]
    for i, (adapt_task, index) in enumerate(zip(adapt_train_tasks, conf.adapt_task_indices)):
        adapt_data_path = os.path.join(DATA_DIR, 
                                       adapt_task.task_name, 
                                       f'{conf.benchmark_type}-0', str(index))
        offline_buffer = np.load(os.path.join(adapt_data_path, 'expert.npz'))
        offline_buffer = SampleBatch(
                obs=offline_buffer['obs'] ,
                obs_next=offline_buffer['obs_next'],
                act=offline_buffer['act'],
                rew=offline_buffer['rew'],
                done=offline_buffer['done'],
        )
        adapt_task.data_buffer = data_preprocessing_from_transition_buffer(offline_buffer, 
            using_transformer=True, segment_length=conf.test_buffer_size)
        adapt_task.data_buffer['label'] = np.ones((adapt_task.data_buffer.context.shape[0], 1)) * i
    sac_rewards = []
    for j, (test_task, index) in enumerate(zip(meta_test_tasks, conf.test_task_indices)):
        test_data_path = os.path.join(DATA_DIR, test_task.task_name, 
                                      f'{conf.benchmark_type}-1', str(index))
        offline_buffer = np.load(os.path.join(test_data_path, 'expert.npz'))
        offline_buffer = SampleBatch(
                obs=offline_buffer['obs'] ,
                obs_next=offline_buffer['obs_next'],
                act=offline_buffer['act'],
                rew=offline_buffer['rew'],
                done=offline_buffer['done'],
        )
        test_task.data_buffer = data_preprocessing_from_transition_buffer(offline_buffer,
            using_transformer=True, segment_length=conf.test_buffer_size, one_context=True)
        logger.info(f"{test_task.task_name}-{j}: {test_task.data_buffer['rew'].mean() * 120}")
        sac_rewards.append(test_task.data_buffer['rew'].mean() * 120)
        test_task.data_buffer = SampleBatch(test_task.data_buffer[0:conf.test_buffer_size])
        logger.info(f"{test_task.task_name}-{j}: {test_task.data_buffer['rew']} ")
    logger.info(f"Test_Mean_Env_Reward-SAC: {np.mean(sac_rewards)}")

    # 5. set log
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    conf['save_path'] = os.path.join(
            BASE_DIR,
            f'log/meta_learning/{conf.benchmark_type}/{conf.policy_type}/{conf.encoder_type}_{conf.loss_type}' if conf.policy_type=='vector'
            else f'log/meta_learning/{conf.benchmark_type}/{conf.policy_type}',
            f'{conf.task_name}' if conf.benchmark_type == 'ml1' else 'multi-tasks',
            f'seed{args.seed}')
    log = SummaryWriter(log_dir=os.path.join(conf.save_path, cur_time))
    ws = WeightSearch(base_policy_list, adapt_train_tasks, meta_test_tasks, conf, device, log)
    ws.meta_training()


if __name__ == "__main__":
    frapo_train()
