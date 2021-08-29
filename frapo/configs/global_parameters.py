import os
BASE_DIR = '/home/iwbyyyy/ubuntu/paper_code/'
DATA_DIR = os.path.join(BASE_DIR, 'offline_data')
SAC_POLICY_DIR = os.path.join(BASE_DIR, 'sac_model')
OFFLINE_POLICY_DIR = os.path.join(BASE_DIR, 'offline_model')
for item in [DATA_DIR, SAC_POLICY_DIR, OFFLINE_POLICY_DIR]:
    if not os.path.exists(item):
        os.makedirs(item)
