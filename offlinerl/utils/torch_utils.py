import uuid
import os
import random
import numpy as np
import torch 
import torch.nn as nn
from torch.functional import F

def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def soft_sync(targ_model: nn.Module, model: nn.Module, tau: float) -> None:
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)
            
def to_gpu(obj, 
           device_id: int = 0):
    device = "cuda:"+str(device_id)
    for key in dir(obj):
        module = getattr(obj, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.to(device)
            
def to_cpu(obj):
    for key in dir(obj):
        module = getattr(obj, key)
        if isinstance(module, (torch.nn.Module, torch.nn.Parameter)):
            module.to("cpu")

def select_free_cuda():
    # 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
    tmp_name = str(uuid.uuid1()).replace("-","")
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >'+tmp_name)
    memory_gpu = [int(x.split()[2]) for x in open(tmp_name, 'r').readlines()]
    os.system('rm '+tmp_name)  # 删除临时生成的 tmp 文件
    
    return np.argmax(memory_gpu)

def get_free_device_fn():
    device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'

    return device

        
def nth_derivative(f, wrt, n):
    for i in range(n):
        if not f.requires_grad:
            return torch.zeros_like(wrt)
        grads = torch.autograd.grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


if __name__ == "__main__":
    print(select_free_cuda())