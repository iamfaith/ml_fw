
from default_transform import *
from rules_transform import *


def get_transform(config, dataset_type, **params):
    if not config['transform']:
        f = globals().get(config['framework']+'_default')
    else:
        f = globals().get(config['framework']+"_" +
                          config['transform']['name'])
    print(f, config['framework']+"_" +
          config['transform']['name'], params)
    if params is not None:
        return f(config, dataset_type, **params)
    else:
        return f(config, dataset_type)  # , **config.transform.params)


if __name__ == "__main__":
    import sys, os
    abs_path = os.path.abspath('./')
    sys.path.append(abs_path)
    import init
    config = init.cfg
    # load_config(
    #     '/Users/faith/Desktop/pytorch/config/example.policy.yml')
    print(config)
    get_transform(config, 'train')  # , rules='./transform/rules.json')
