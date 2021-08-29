import torch
import os
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm
from frapo.configs.global_parameters import BASE_DIR, DATA_DIR
from offlinerl.task import load_ml1_tasks
from torch.utils.tensorboard import SummaryWriter
from offlinerl.utils.models.builders import (create_transformer_classification_model,
    create_lstm_classification_model, create_transformer_for_rl)
from offlinerl.utils.conf_utils import get_default_hparams, deep_update_dict
from offlinerl.utils.torch_utils import select_free_cuda, setup_seed
from offlinerl.utils.dataset import (SampleBatch, data_preprocessing_from_transition_buffer)


class ContextEncoderTraining:
    def __init__(self, train_tasks, observation_shape, action_size, hparams, device, log):
        self.hparams = hparams
        self.device = device
        self.log = log
        self.train_tasks = train_tasks
        if self.hparams.encoder_type == 'transformer':
            self.context_encoder = create_transformer_classification_model(observation_shape, action_size, self.hparams.encoder_size,
                hparams.sequence_length, len(hparams.task_indices), device).to(device)
        elif self.hparams.encoder_type == 'lstm':
            self.context_encoder = create_lstm_classification_model(
                observation_shape,
                action_size,
                self.hparams.encoder_size,
                len(hparams.task_indices),
                device,
            ).to(device)
        elif self.hparams.encoder_type == 'transformer_rl':
            self.context_encoder = create_transformer_for_rl(
                observation_shape, 
                action_size,
                self.hparams.encoder_size,
                self.hparams.sequence_length, 
                device
            ).to(device)
        else:
            raise NotImplementedError('No such model!')

        self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=self.hparams.learning_rate)
    
    def encoder_loss_func(self, batch_i, batch_j): # ses
        batch_i.to_torch(dtype=torch.float32, device=self.device)
        batch_j.to_torch(dtype=torch.float32, device=self.device)
        label_i, label_j = batch_i['label'], batch_j['label']
        context_i, context_j = batch_i['context'], batch_j['context']
        embedding_i = self.context_encoder.get_encoder_vector(context_i)
        embedding_j = self.context_encoder.get_encoder_vector(context_j)
        if self.hparams.loss_type == 'negt': # negative-power variant of contrastive loss
            same_class_loss = (label_i == label_j) * ((embedding_i - embedding_j)**2).sum(-1, keepdim=True)
            different_class_loss = ~(label_i == label_j) * self.hparams.beta * 1 / (((embedding_i - embedding_j)**2).sum(-1, keepdim=True) + 1e-10)
            metric_loss = same_class_loss + different_class_loss
        elif self.hparams.loss_type == 'cont': # contrasitive loss
            same_class_loss = (label_i == label_j) * ((embedding_i - embedding_j)**2).sum(-1, keepdim=True)
            tmp_tensor = self.hparams.contrasitive_m - torch.sqrt(((embedding_i - embedding_j)**2).sum(-1, keepdim=True) + 1e-10)
            different_class_loss = ~(label_i == label_j) * \
                    (torch.maximum(torch.zeros((tmp_tensor.shape[0], 1), device=self.device), tmp_tensor)**2)
            metric_loss = same_class_loss + \
                ~(label_i == label_j) * \
                    (torch.maximum(torch.zeros((tmp_tensor.shape[0], 1), device=self.device), tmp_tensor)**2)
        else:
            raise NotImplementedError()

        return metric_loss.mean(), same_class_loss.detach().mean().item(), different_class_loss.detach().mean().item()
    
    def encoder_training(self):
        self.context_encoder.train()
        total_batch = 0
        val_best_loss = np.inf
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升

        for batch_idx in tqdm(range(self.hparams.max_steps)):
            encoder_loss = 0.0
            same_class_loss_list = []
            different_class_loss_list = []
            for index in np.arange(len(self.train_tasks)):
                    task = self.train_tasks[index]
                    batch = task.train_buffer.sample(self.hparams.batch_size)
                    for index_extra in np.arange(len(self.train_tasks)):
                        task_extra = self.train_tasks[index_extra]
                        batch_extra = task_extra.train_buffer.sample(self.hparams.batch_size)
                        total_loss, same_class_loss_value, different_class_loss_value = self.encoder_loss_func(batch, batch_extra)
                        encoder_loss += total_loss
                        if index == index_extra:
                            same_class_loss_list.append(same_class_loss_value)
                        else:
                            different_class_loss_list.append(different_class_loss_value)
            self.context_encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.context_encoder_optimizer.step()

            if batch_idx % self.hparams.eval_interval==0:
                val_same_loss, val_different_loss = self.evaluate()
                val_loss = val_same_loss + val_different_loss
                train_loss = np.array(same_class_loss_list).mean() + np.array(different_class_loss_list).mean()
                self.log.add_scalar("loss/train", train_loss.item(), total_batch)
                self.log.add_scalar("loss/train_same_loss", np.array(same_class_loss_list).mean(), total_batch)
                self.log.add_scalar("loss/train_different_loss", np.array(different_class_loss_list).mean(), total_batch)
                self.log.add_scalar("loss/val", val_loss.item(), total_batch)
                self.log.add_scalar("loss/val_same_loss", val_same_loss, total_batch)
                self.log.add_scalar("loss/val_different_loss", val_different_loss, total_batch)
                self.context_encoder.train()
            
                if val_loss < val_best_loss:
                    val_best_loss = val_loss
                    logger.info(f'update model in {total_batch}')
                    torch.save(self.context_encoder.state_dict(), os.path.join(self.hparams.save_path, 
                                                                               f'encoder_{self.hparams.encoder_type}_{self.hparams.loss_type}.pt'))
                    last_improve = total_batch
    
                logger.info(f'Iter: {total_batch},  Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            total_batch += 1
            if total_batch - last_improve > self.hparams.require_improvement:
                # 验证集metric超过一定batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            return

    def evaluate(self, eval_time=5):
        self.context_encoder.eval()
        same_class_loss_list = []
        different_class_loss_list = []
        for _ in range(eval_time):
            for index in np.arange(len(self.train_tasks)):
                    task = self.train_tasks[index]
                    batch = task.val_buffer.sample(self.hparams.batch_size)
                    for index_extra in np.arange(len(self.train_tasks)):
                        task_extra = self.train_tasks[index_extra]
                        batch_extra = task_extra.val_buffer.sample(self.hparams.batch_size)
                        with torch.no_grad():
                            _, same_class_loss_value, different_class_loss_value = self.encoder_loss_func(batch, batch_extra)
                            if index == index_extra:
                                same_class_loss_list.append(same_class_loss_value)
                            else:
                                different_class_loss_list.append(different_class_loss_value)
        self.context_encoder.train()
        return np.array(same_class_loss_list).mean(), np.array(different_class_loss_list).mean()


def metric_learning_of_encoder():
    import argparse
    parser = argparse.ArgumentParser('Train an encoder with metric learning!')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_path', type=str, default=None)
    args = parser.parse_args()
    args.gpu = select_free_cuda()
    device = torch.device(f'cuda:{args.gpu}')
    setup_seed(args.seed)

    default_conf = get_default_hparams(OmegaConf.load(os.path.join(
                BASE_DIR,
                'frapo/configs/encoder/metric_learning/default.yaml'
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
    if conf.benchmark_type == 'ml1':
        tasks = load_ml1_tasks(conf.task_name)[0]
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    
    train_tasks = [tasks[index] for index in conf.task_indices]

    # 1. load the data of the train_tasks
    observation_shape, action_size = train_tasks[0].observation_shape, train_tasks[0].action_size
    for i, (train_task, index) in enumerate(zip(train_tasks, conf.task_indices)):
        if conf.benchmark_type == 'ml1':
            data_path = os.path.join(DATA_DIR, train_task.task_name, 
                                    f'{conf.benchmark_type}-0', str(index))
            offline_buffer = np.load(os.path.join(data_path, 'expert.npz'))
            offline_buffer = SampleBatch(
                    obs=offline_buffer['obs'] ,
                    obs_next=offline_buffer['obs_next'],
                    act=offline_buffer['act'],
                    rew=offline_buffer['rew'],
                    done=offline_buffer['done'],
            )
        else:
            raise ValueError('Not implemented!')
        train_task.data_buffer = data_preprocessing_from_transition_buffer(offline_buffer, 
            using_transformer=True, segment_length=conf.sequence_length)
        n = train_task.data_buffer.context.shape[0]
        train_task.data_buffer['label'] = np.ones((n, 1)) * i
        train_task.train_buffer = SampleBatch(train_task.data_buffer[0:int(n * conf.train_data_ratio)])
        train_task.val_buffer = SampleBatch(train_task.data_buffer[int(n * conf.train_data_ratio):])
 
    if conf.benchmark_type == 'ml1':
        conf.save_path = os.path.join(
                BASE_DIR,
                f'log/encoder_learning/{conf.benchmark_type}/{conf.task_name}/seed{args.seed}/',
                conf.encoder_type, conf.loss_type)
    else:
        raise ValueError('Not implemented!')
    log = SummaryWriter(log_dir=conf.save_path)
    cet = ContextEncoderTraining(train_tasks, observation_shape, action_size, conf, device, log)
    cet.encoder_training()


if __name__ == "__main__":
    metric_learning_of_encoder()