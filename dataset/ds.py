import torch
from torch.utils.data import DataLoader


def _get_param(config, param, default=None):
    if config['data'] and param in config['data']:
        return config['data'][param]
    return default


def load_dataset(config, x, y=None, transformer=None):
    if not config['data']:
        f = globals().get(config['framework']+'DefaultDataset')
    else:
        f = globals().get(config['framework']+"_" +
                          config['transform']['name'])
    if f:
        return f(config, x, y, transformer)


def get_dataloader(config, dataset_type, x, y=None,  transform=None, **_):
    assert config['train']['batch_size']
    dataset = load_dataset(config.data, x, y, transform)

    is_train = 'train' == dataset_type
    # if is_train else config.eval.batch_size
    batch_size = config['train']['batch_size']
    num_workers = _get_param(config, 'num_workers', 4)

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            drop_last=is_train,
                            num_workers=num_workers,  # Multi-process data loading
                            pin_memory=True)  # https://pytorch.org/docs/stable/data.html
    return dataloader


if __name__ == "__main__":
    import sys
    import os
    abs_path = os.path.abspath('./')
    sys.path.append(abs_path)
    import init
    config = init.cfg
    # load_config(
    #     '/Users/faith/Desktop/pytorch/config/example.policy.yml')
    print(config)
