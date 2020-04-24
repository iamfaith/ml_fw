

# from common import datatype


def keras_default(config, dataset_type):

    from keras.preprocessing.image import ImageDataGenerator
    if dataset_type == 'train':
        return ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
    else:
        return ImageDataGenerator(rescale=1./255)


def torch_default(config, dataset_type):
    import torchvision.transforms as transforms
    if dataset_type == 'train':
        return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉圖片
        transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    ])
    else:
        return transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])