import itertools
import numpy as np
from albumentations import Compose, RandomRotate90, Flip, Transpose, Resize
from albumentations import RandomContrast, RandomBrightness, RandomGamma
from albumentations import Blur, MotionBlur, InvertImg
from albumentations import Rotate, ShiftScaleRotate, RandomScale
from albumentations import GridDistortion, ElasticTransform
import random


def _check_param(config, param):
    if config['transform']['params'] and param in config['transform']['params']:
        return config['transform']['params'][param]
    return None


def keras_rules(config, dataset_type, **params):
    print('torch_rules', params)
    assert 'rules' in params or config['transform']['rules']

    size = _check_param(config, 'size')
    size = 128 if size is None else size

    if 'rules' in params:
        rules_file = params['rules']
    else:
        rules_file = config['transform']['rules']
    with open(rules_file, 'r') as fid:
        rules = eval(fid.read())
        rules = itertools.chain.from_iterable(rules)
        print(rules)
    means = np.array([127.5, 127.5, 127.5, 127.5])
    stds = np.array([255.0, 255.0, 255.0, 255.0])

    base_aug = Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
    ])
    aug_list = []
    for rule in rules:

        op_1, params_1 = rule[0]
        op_2, params_2 = rule[1]
        print(op_1, params_1)
        aug = Compose([
            globals().get(op_1)(**params_1),
            globals().get(op_2)(**params_2),
        ])
        aug_list.append(aug)
    print('len(aug_list):', len(aug_list), 'image size', size)
    resize = Resize(height=size, width=size, always_apply=True)

    def transform(image):
        if dataset_type == 'train':
            image = base_aug(image=image)['image']
            if len(aug_list) > 0:
                aug = random.choice(aug_list)
                image = aug(image=image)['image']
            image = resize(image=image)['image']
        else:
            if size != image.shape[0]:
                image = resize(image=image)['image']

        image = image.astype(np.float32)
        if  _check_param(config, 'per_image_norm'):
            mean = np.mean(image.reshape(-1, 4), axis=0)
            std = np.std(image.reshape(-1, 4), axis=0)
            image -= mean
            image /= (std + 0.0000001)
        else:
            image -= means
            image /= stds
        image = np.transpose(image, (2, 0, 1))

        return image

    return transform
