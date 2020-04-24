import glob
import cv2
import math
import matplotlib.pyplot as plt
import functools
import yaml
import os
import collections


class ImgUtil(object):

    def __init__(self, path=None, imgs=None):
        super().__init__()
        self.path = path
        self.imgs = imgs
        if self.path is not None:
            self.imgs = glob.glob(self.path)

    def show(self, end, begin=0):
        assert begin <= end
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=0, hspace=0)
        sub = int(round(math.sqrt(end)))
        for index, img in enumerate(self.imgs[begin:end]):
            img = cv2.imread(img)
            img = cv2.resize(img, (250, 250))
            plt.axis('off')
            plt.subplot(sub, sub, index+1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class Decorator(object):
    @staticmethod
    def log(is_debug=False):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                if is_debug:
                    print('call %s():' % func)
                    print('parameters:', args, kw)
                return func(*args, **kw)

            return wrapper

        return decorator


def load_config(filepath, Loader=yaml.Loader):
    assert os.path.exists(filepath)
    with open(filepath) as fid:
        config = yaml.load(fid)
        # print(config['optimizer'])
    return collections.defaultdict(str, config)
# Visualize images in the dataset
# characters = glob.glob('simpsons-dataset/kaggle_simpson_testset/kaggle_simpson_testset/**')

# for character in characters[:25]:
#     img = cv2.imread(character)
#     print(img.shape)
#     img = cv2.resize(img, (250, 250))
#     print('--after', img.shape)
#     plt.axis('off')
#     plt.subplot(5, 5, i+1) #.set_title(l)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     i += 1


if __name__ == "__main__":
    print(load_config('/Users/faith/Desktop/pytorch/config/example.policy.yml'))
    # img = ImgUtil()
    # img.show(24)
