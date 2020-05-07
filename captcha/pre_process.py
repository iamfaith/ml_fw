from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import cv2
import glob
import math
import torch
from torchvision.models.resnet import ResNet, BasicBlock

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


class IterableDataset(IterableDataset):

    def __init__(self, filepath, transform=None, decoder=None, width=120, height=50, has_label=True):
        # super().__init__()
        self.filepath = filepath
        # img_size = (50, 120)
        self.width, self.height = width, height
        self.transform = transform
        self.decoder = decoder
        self.has_label = has_label
    # def parseFile(self, filepath):
    #     with open(filepath, 'r') as f:
    #         for line in f:

    #             token = line.strip('\n').strip(' ')
    #             print('-----', line, token)
    #             yield from token

    # def get_stream(self, filepath):
        # from itertools import cycle
        # return self.parseFile(filepath)

    def read_img(self, img_dir):
        # print(img_dir)
        for img in img_dir:

            img_gray = Image.open(img).convert('L')
            img_two = img_gray.point(lambda x: 0 if x > 129 else 255)

            x = cv2.resize(
                np.array(img_two), (self.width, self.height))

            # x = cv2.resize(cv2.imread(img), (self.width, self.height))
            # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

            if self.transform is not None:
                x = self.transform(x)



            pos, temp = [], []
            tmp = x.T  # .squeeze(0).numpy()
            for i, col in enumerate(tmp):
                if np.sum(col) > 0:
                    # print(i)
                    if len(pos) == 0:
                        pos.append(i)
                        temp.append(i)
                    else:
                        if temp[len(temp) - 1] + 1 != i:
                            pos.append(i)
                        temp.append(i)
            
            # print('-----', img, pos, len(y), x.shape)

            if len(pos) == 4:
                if self.has_label:
                    y = [self.decoder(i) if self.decoder is not None else i for i in img[-8:-4].lower()]
                for i, p in enumerate(pos):
                    if i < 3:
                        end = pos[i+ 1] - 1
                        if i == 0:
                            start = 0
                        else:
                            start = pos[i] - 1
                    else:
                        start, end = pos[i] - 1, self.width - 1
                    
                    # start, end = pos[i * 2], pos[i * 2 + 1]
                    # print(i, start, end)
                    if self.has_label:
                        charater =cv2.resize(x[:, start:end], (50, 50))
                        yield torch.FloatTensor([charater]), keys.get(y[i])
                    else:
                        charater =cv2.resize(x[:, start + 5:end -5 ], (50, 50))
                        yield torch.FloatTensor([charater])
            else:
                os.rename(img, '/Users/faith/Downloads/captcha-dataset/unreg2/' + os.path.basename(img))
                # print(img)
            # yield x, y

    def __iter__(self):
        return self.read_img(self.filepath)


def debug(one_channel):
    global isdebug
    if not isdebug:
        return
    # print(type(one_channel))
    if isinstance(one_channel, torch.Tensor):
        # print(one_channel.size())
        one_channel = one_channel.squeeze(0).numpy()
        # print(one_channel.shape)
        three_channel = np.array(
            [one_channel, one_channel, one_channel]).transpose(1, 2, 0)
    else:
        three_channel = np.array(
            [one_channel, one_channel, one_channel]).transpose(1, 2, 0)

    # print('---debug',three_channel.shape)
    cv2.imshow('img', three_channel)
    cv2.waitKey(0)



def train_val_split(path, batch_size=64, decoder=None):
    samples = glob.glob(path)
    np.random.shuffle(samples)
    nb_train = math.ceil(0.9 * len(samples))  # 共有10万+样本，9万用于训练，1万+用于验证
    train_samples = samples[:nb_train]
    train_dataset = IterableDataset(
        train_samples, transform=None, decoder=decoder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    val_samples = samples[nb_train:]

    val_dataset = IterableDataset(val_samples, transform=None, decoder=decoder)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, train_samples, val_samples


    

letter_list = [chr(i) for i in range(97,123)]
char_list =  [str(i) for i in range(0, 10)] + letter_list

keys = {}
values = {}
for i, c in enumerate(char_list):
    keys[c] = i
    values[i] = c

isdebug = True
train_loader, val_loader, train_samples, val_samples = train_val_split(r'/Users/faith/Downloads/captcha-dataset/label/*png', batch_size=64)

for i, data in enumerate(train_loader):
    print(i, data[0].size())
 
    # debug(data[0][0])
    break


print( len(train_samples))
# pre_process(r'/Users/faith/Downloads/captcha-dataset/label/*png')



