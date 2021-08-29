### Introduction

The project consists of mainly two parts currently：

- offlinerl: the offline reinforcement learning library implemented based on [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning). To know how to use the implemented algorithms, you should be familiar with it.
- frapo: the algorithm **frapo** based on the offlinerl library which achieves a meta offline reinforcement learning procedure. To understand offline meta RL and our work, please refer to this [material](http://proceedings.mlr.press/v139/mitchell21a/mitchell21a.pdf).

### Installation

To install locally, you should firstly install [MuJoCo](https://github.com/openai/mujoco-py) and [MetaWorld](https://github.com/rlworkgroup/metaworld). For the remaining dependencies, create conda environment by running:

```shell
conda env create -f rl_base.yaml
conda activate rl_base
```

After create the conda environment, you can install the package **metarl** by running:

```sh
pip install -e .
```


### Project structure and explanation
The current code organization for the project is as follows:

```
xRL
├── xRL/frapo
│   ├── xRL/frapo/configs    # the configuration files for frapo 
│   ├── xRL/frapo/encoder    # the api for training a task encoder
│   └── xRL/frapo/offline    # the api for online training、sampling offline data、training offline policy
├── xRL/frapo/meta_learning.py    # the main entrance for offline meta learning
├── xRL/offlinerl
│   ├── xRL/offlinerl/algos    # the implemented algorithms in this library
│   │   ├── xRL/offlinerl/algos/dynamics
│   │   └── xRL/offlinerl/algos/modelfree
│   ├── xRL/offlinerl/config    # the configuration directory for the algorithms
│   │   ├── xRL/offlinerl/config/algos
│   │   └── xRL/offlinerl/config/logger
│   ├── xRL/offlinerl/evaluation    # the evalution api for the policy
│   ├── xRL/offlinerl/task    # the api for load the benchmark, such as d4RL、MetaWorld
│   ├── xRL/offlinerl/trainer    # the implementation of a trainer
│   └── xRL/offlinerl/utils   # the implementation of the neural network and some tool functions
│       └── xRL/offlinerl/utils/models
│           └── xRL/offlinerl/utils/models/torch
└── xRL/README.md
```
