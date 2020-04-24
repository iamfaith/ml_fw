
import sys
import os
import yaml
import collections


def load_config(filepath, Loader=yaml.Loader):
    assert os.path.exists(filepath)
    with open(filepath) as fid:
        config = yaml.load(fid)
        assert 'framework' in config and 'project' in config
        # print(config['optimizer'])
    return collections.defaultdict(str, config)


def init_cfg(filepath):
    assert os.path.exists(filepath)
    global cfg
    cfg = load_config(filepath)
    print('from', filepath, 'load config', cfg)
    if cfg["wandb"]:
        import wandb
        wandb.init(project=cfg['project'])
        config = wandb.config
        config.batch_size = cfg['train']['batch_size']
        config.epochs = cfg['train']['epochs']
        config.learn_rate = cfg['train']['learning_rate']
    return cfg


if __name__ == "__main__":
    abs_path = os.path.abspath('./')
    init_cfg(os.path.join(abs_path + '/config', 'test.policy.yml'))
    global cfg
    print(cfg)
