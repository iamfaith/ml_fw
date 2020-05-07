import glob
import cv2
import math
import matplotlib.pyplot as plt
import functools
import os


class ImgUtil(object):

    def __init__(self, path=None, imgs=None):
        super().__init__()
        self.path = path
        self.imgs = imgs
        if self.path is not None:
            assert os.path.exists(self.path)
            self.imgs = os.listdir(self.path)

    def show(self, end, begin=0):
        assert begin <= end
        print(self.imgs[begin:end])
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=0, hspace=0)
        sub = int(round(math.sqrt(end)))
        for index, img in enumerate(self.imgs[begin:end]):
            img = cv2.imread(os.path.join(self.path, img))
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


def readfile(path, label, height=299, width=299):
    assert os.path.exists(path)

    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), height, width, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (height, width))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x


def torchIterableDataset():
    from torch.utils.data import Dataset, IterableDataset, DataLoader
    import torchvision.transforms as transforms
    import numpy as np
    from PIL import Image
    import os
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

        def __init__(self, filepath, transform=None):
            # super().__init__()
            self.filepath = filepath
            img_size = (50, 120)
            self.width, self.height = img_size[1], img_size[0]
            self.transform = transform

        def parseFile(self, filepath):
            with open(filepath, 'r') as f:
                for line in f:

                    token = line.strip('\n').strip(' ')
                    print('-----', line, token)
                    yield from token

        # def get_stream(self, filepath):
            # from itertools import cycle
            # return self.parseFile(filepath)

        def read_img(self, img_dir):
            # print(img_dir)
            for img in img_dir:
                print(img)
                img_gray = Image.open(img).convert('L')
                img_two = img_gray.point(lambda x: 255 if x > 129 else 0)
                 
                one_channel = cv2.resize(np.array(img_two), (self.width, self.height)) 
                x = np.array([one_channel, one_channel, one_channel]).transpose(1,2,0)
                
                #x = cv2.resize(cv2.imread(img), (self.width, self.height))

                if self.transform is not None:
                    x = self.transform(x)

                # print(numpy.array(x).transpose(1,2 , 0).shape)
                # cv2.imshow('new', numpy.array(x).transpose(1,2 , 0))
                # cv2.waitKey(0)
                y = [keys.get(i) for i in img[-8:-4].lower()]

                # print('---', x,y)
                yield x, np.array(y)

        def __iter__(self):
            return self.read_img(self.filepath)

    samples = glob.glob(r'/Users/faith/Downloads/captcha-dataset/label/*png')
    
    for s in samples:
        if len(os.path.basename(s)) != 8:
            print(s)
            os.remove(s)
        
    
    
    np.random.shuffle(samples)
    nb_train = math.ceil(0.9 * len(samples))  # 共有10万+样本，9万用于训练，1万+用于验证
    train_samples = samples[:nb_train]
    test_samples = samples[nb_train:]

    train_dataset = IterableDataset(train_samples, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=2)
    print(train_loader)
    letter_list = [chr(i) for i in range(97,123)]
    char_list =  [str(i) for i in range(0, 10)] + letter_list

    keys = {}
    values = {}
    for i, c in enumerate(char_list):
        keys[c] = i
        values[i] = c
    for i, data in enumerate(train_loader):
        print(i, data)
        break
    # test_dataset = IterableDataset(test_samples, transform=test_transform)
    # test_loader = DataLoader(test_dataset, batch_size=100)
    
    import pretrainedmodels
    import torch.nn as nn
    class CaptchaModel(nn.Module):
        def __init__(self, num_classes=len(keys)):
            super(CaptchaModel, self).__init__()
            model_name = 'xception'
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained='imagenet')
            conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(in_channels=3,
                                    out_channels=conv1.out_channels,
                                    kernel_size=conv1.kernel_size,
                                    stride=conv1.stride,
                                    padding=conv1.padding,
                                    bias=conv1.bias)

            # copy pretrained weights
            self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
            self.model.conv1.weight.data[:, 3:, :,
                                    :] = conv1.weight.data[:, :1, :, :]

            self.model.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(in_features, num_classes)

        def forward(self, x):
            return self.model(x), self.model(x), self.model(x), self.model(x)

    # model = CaptchaModel(len(keys)).cuda()
    # loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # num_epoch = 3

    # for epoch in range(num_epoch):
    #     epoch_start_time = time.time()
    #     train_acc = 0.0
    #     train_loss = 0.0
    #     val_acc = 0.0
    #     val_loss = 0.0

    #     model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    #     for i, data in enumerate(train_loader):
    #         # print(data[0].shape, data[1].shape, data[1].numpy())
    #         optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
    #         train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
    #         batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
    #         batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
    #         optimizer.step() # 以 optimizer 用 gradient 更新參數值
    #         # print(train_pred.data.numpy(), '-----')
    #         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    #         train_loss += batch_loss.item()
        
    #     model.eval()
    #     with torch.no_grad():
    #         for i, data in enumerate(val_loader):
    #             val_pred = model(data[0].cuda())
    #             batch_loss = loss(val_pred, data[1].cuda())

    #             val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
    #             val_loss += batch_loss.item()

    #         #將結果 print 出來
    #         print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
    #             (epoch + 1, num_epoch, time.time()-epoch_start_time, \
    #             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
            
    
           

    # print(next(enumerate(loader)))
    # for i, data in enumerate(loader):
    #     # print(i, data)
    #     if i > 1:
    #         break


if __name__ == "__main__":
    
    # a = [ [i] for i in range(4) ]
    # print(a, a[0])
    torchIterableDataset()
    # print(load_config('/Users/faith/Desktop/pytorch/config/example.policy.yml'))
    # img = ImgUtil()
    # img.show(24)
