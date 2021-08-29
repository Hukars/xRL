import os
import copy

from omegaconf import OmegaConf

from offlinerl.config import algos as algos_conf


def get_algo_conf(algo_name):    
    algo_module_path = os.path.dirname(algos_conf.__file__)
    conf_file_list = [i for i in os.listdir(algo_module_path) if i.endswith(".yaml")]
    conf_file_name = algo_name.lower() + ".yaml"
    assert conf_file_name in conf_file_list, "Lack of algorithm profile {}".format(conf_file_name)
    
    conf = load_conf(os.path.join(algo_module_path,conf_file_name))
    
    return conf

def get_default_hparams(conf):
    conf = copy_conf(conf)
    for k,v in conf.items():
        conf[k] = v["default"]
        if v["default"] == "None":
            conf[k] = None

    return obj_to_conf(conf)

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

# Check and transform to dot-notation key.
def transform_dot_notation_key(key):
    if not isinstance(key,str):
        if isinstance(key,list):
            key = ".".join(key)
        elif isinstance(key,tuple):
            key = ".".join(key)
        else:
            raise NotImplementedError
        
    return key

# Test if an object is an OmegaConf obejct.
def is_conf(obj):
    return OmegaConf.is_config(obj)

# Show conf.
def show_conf(conf):
    print(OmegaConf.to_yaml(conf))

# Load conf from yaml file.
def load_conf(conf_path):
    conf = OmegaConf.load(conf_path)

    return conf

# Save conf as a yaml file.
def save_conf(conf, conf_path):
    with open(conf_path, "w") as fp:
        OmegaConf.save(config=conf, f=fp.name)

# Get conf from list or dict
def obj_to_conf(obj):
    return OmegaConf.create(obj)

# Merging configurations.
def merge_conf(*confs):
    conf = OmegaConf.merge(*confs)

    return conf

# Tests if a value is missing in conf.
def not_missing_in_conf(conf):
    pass

# Select value from conf by key.
def select_value(conf, key):
    key = transform_dot_notation_key(key)
    value = OmegaConf.select(conf,
                             key,
                             throw_on_missing=True
                            )

    return value

# Update values in your config using a dot-notation key.
def update_conf_(conf, key, value, merge=False):
    key = transform_dot_notation_key(key)
    OmegaConf.update(conf, key, value, merge)

    return conf

def update_conf(conf, key, value, merge=False):
    conf = copy.deepcopy(conf)
    conf = update_conf_(conf, key, value, merge)

    return conf

# Get a mask copy.
def masked_copy(conf, key_list):
    return OmegaConf.masked_copy(conf, key_list)

def copy_conf(conf):
    d = dict(conf)
    return obj_to_conf(d)