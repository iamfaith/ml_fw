
import os

_subffix = {"keras": '.hdf5', "torch": '.pth'}


def wandb(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func)
        return func(*args, **kw)
    return wrapper


def _checkpoint_filename(config):
    return config['framework'] + "-{epoch:04d}" + _subffix.get(config['framework'])


def _load_checkpoint_path(config):
    checkpoint_dir = os.path.join(os.path.abspath(
        config['base_path']), config['project'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, _checkpoint_filename(config))
    # print(checkpoint_path, os.path.exists(checkpoint_path))
    return checkpoint_path


def _torch_wandb_checkpoint(config, weights):
    import wandb
    torch.save(weights, os.path.join(
        wandb.run.dir, _checkpoint_filename(config)))


def save_torch_checkpoint(config, model, optimizer, epoch, step, weights_dict=None, name=None):
    checkpoint_path = _load_checkpoint_path(config)
    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }
    import torch
    torch.save(weights_dict, checkpoint_path)
    if config["wandb"]:
        _torch_wandb_checkpoint(config, weights_dict)


def _keras_wandb_checkpoint(config):
    from wandb.keras import WandbCallback
    return WandbCallback()


def save_keras_checkpoint(config):
    checkpoint_path = _load_checkpoint_path(config)
    print(checkpoint_path)
    cp = []
    if config["wandb"]:
        print("add wandb")
        cp.append(_keras_wandb_checkpoint(config))
    from keras.callbacks import ModelCheckpoint
    keras_cp = ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_best_only=True, monitor='loss')
    cp.append(keras_cp)
    return cp


def save_checkpoint(config):
    assert 'framework' in config and 'project' in config

    print('config:', config)
    f = globals().get('save_{}_checkpoint'.format(config['framework']))

    return f(config)
    # if config.model.params is None:
    #     return f()
    # else:
    #     return f(**config.model.params)


if __name__ == "__main__":
    import sys
    abs_path = os.path.abspath('./')
    print(abs_path)
    sys.path.append(abs_path)
    from common import load_config
    config = load_config(
        '/Users/faith/Desktop/pytorch/config/example.policy.yml')
    print(config, config)
    print(save_checkpoint(config))
