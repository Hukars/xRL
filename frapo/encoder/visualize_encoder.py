import os
import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import rgb2hex
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE
from frapo.configs.global_parameters import BASE_DIR, DATA_DIR
from offlinerl.task import load_ml1_tasks
from offlinerl.utils.conf_utils import get_default_hparams
from offlinerl.utils.torch_utils import setup_seed, select_free_cuda
from offlinerl.utils.models.builders import (create_transformer_classification_model,
    create_lstm_classification_model, create_transformer_for_rl)
from offlinerl.utils.dataset import data_preprocessing_from_transition_buffer, SampleBatch


def draw_tsne_pict(X, y, conf, seed, algo='FOCAL'):
    X_std = StandardScaler().fit_transform(X) 
    tsne = TSNE(n_components=2) 
    X_tsne = tsne.fit_transform(X_std) 
    X_tsne_data = np.concatenate([X_tsne, y], 1)
    df_tsne = pd.DataFrame(X_tsne_data, columns=["Dim1", "Dim2", "class"]) 

    colors = tuple([(torch.sigmoid(torch.randn(1)).item(),torch.sigmoid(torch.randn(1)).item(), 
        torch.sigmoid(torch.randn(1)).item()) for i in range(len(conf.task_indices))])
    colors = [rgb2hex(x) for x in colors]
    # black, gray, red, yellow, green, aqua, blue, purple
    colors = ["#000000", "#808080", "#FF0000","#FFFF00",
        "#00FF00", "#00FFFF", "#0000FF", "#800080", "#9FE2BF", "#40E0D0", "#6495ED", "#CCCCFF"]
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    for i, color in enumerate(colors):
        need_idx = np.where(y==i)[0]
        ax.scatter(X_tsne[need_idx, 0],X_tsne[need_idx, 1], c=color, label=i)
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(conf.title)
    if not os.path.exists(os.path.join(BASE_DIR, conf.save_dir)):
        os.makedirs(os.path.join(BASE_DIR, conf.save_dir))
    fig.savefig(os.path.join(BASE_DIR, conf.save_dir, f'{conf.task_name}_{algo}_{seed}.png'))
    plt.savefig(os.path.join(BASE_DIR, conf.save_dir, f'{conf.task_name}_{algo}_{seed}.pdf'), bbox_inches='tight')


def visualize_latent_vector():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='frapo/configs/encoder/visualize/default.yaml')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(f'cuda:{select_free_cuda()}')
    conf = get_default_hparams(OmegaConf.load(os.path.join(
                BASE_DIR,
                args.config_path
            )
        )
    )
    if conf.benchmark_type == 'ml1':
        tasks = load_ml1_tasks(conf.task_name)[0]
    else:
        raise ValueError('Other benchmarks are not supported in this project!')
    train_tasks = [tasks[index] for index in conf.task_indices]
    observation_shape, action_size = train_tasks[0].observation_shape, train_tasks[0].action_size

  # 1. load transformer encoder
    if conf.encoder_path is None:
        raise ValueError('You should supply the path of the encoder in the config file!')
    
    if conf.encoder_type == 'transformer':
        context_encoder = create_transformer_classification_model(
            observation_shape,
            action_size,
            conf.encoder_size,
            conf.sequence_length,
            conf.class_num,
            device,
        ).to(device)
    elif conf.encoder_type == 'lstm':
            context_encoder = create_lstm_classification_model(
                observation_shape,
                action_size,
                conf.encoder_size,
                len(conf.task_indices),
                device,
            ).to(device)
    elif conf.encoder_type == 'transformer_rl':
            context_encoder = create_transformer_for_rl(
                observation_shape, 
                action_size,
                conf.encoder_size,
                conf.sequence_length, 
                device
            ).to(device)
    else:
        raise NotImplementedError('No such model!')

    context_encoder.load_state_dict(torch.load(conf.encoder_path))
    context_encoder.eval()

    #2. load the data of the adapt tasks and test task
    train_data_list, train_label_list = [], []
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
            raise ValueError
        train_task.train_buffer = data_preprocessing_from_transition_buffer(offline_buffer, 
            using_transformer=True, segment_length=conf.sequence_length)
        n = train_task.train_buffer['context'].shape[0]
        perm = np.random.permutation(n)
        index = perm[0:conf.point_num]
        train_data_list.append(train_task.train_buffer['context'][index])
        train_label_list.append(np.ones((conf.point_num, 1), dtype=np.int64) * i)

    input_data = torch.from_numpy(np.concatenate(train_data_list, 0)).float().to(device)
    X = context_encoder.get_encoder_vector(input_data).detach().cpu().numpy()
    y = np.concatenate(train_label_list, 0) 
    draw_tsne_pict(X, y, conf, args.seed, 'FORLAP')


if __name__ == '__main__':
    visualize_latent_vector()