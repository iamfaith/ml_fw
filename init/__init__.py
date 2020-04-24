import sys, os
abs_path = os.path.abspath('./')
# print(abs_path)
sys.path.append(abs_path)
from common import load_config
cfg = load_config(os.path.join(abs_path + '/config', 'example.policy.yml'))
print('from', abs_path, 'load config',cfg)


if cfg["wandb"]:
    import wandb
    wandb.init(project=cfg['project'])
    config = wandb.config
    config.batch_size = cfg['train']['batch_size']
    config.epochs = cfg['train']['epochs']
    config.learn_rate = cfg['train']['learning_rate']
