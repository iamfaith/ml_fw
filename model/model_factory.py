import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import pretrainedmodels


def keras_InceptionV3(num_classes=11):
    from keras.applications.inception_v3 import InceptionV3
    from keras.layers import Flatten, Dense, AveragePooling2D
    from keras.models import Model
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
    # Note that the preprocessing of InceptionV3 is:
    # (x / 255 - 0.5) x 2

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(num_classes, activation='softmax', name='predictions')(output)

    InceptionV3_model = Model(InceptionV3_notop.input, output)
    return InceptionV3_model
    # InceptionV3_model.summary()
    # print(len(InceptionV3_model.layers))
    
    
def get_resnet34(num_classes=28, in_channels=4, **_):
    model_name = 'resnet34'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
    model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_resnet18(num_classes=28, **_):
    model_name = 'resnet18'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.conv1
    model.conv1 = nn.Conv2d(in_channels=4,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
    model.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_senet(model_name='se_resnext50', num_classes=28, **_):
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    conv1 = model.layer0.conv1
    model.layer0.conv1 = nn.Conv2d(in_channels=4,
                                   out_channels=conv1.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding,
                                   bias=conv1.bias)

    # copy pretrained weights
    model.layer0.conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model.layer0.conv1.weight.data[:,3:,:,:] = conv1.weight.data[:,:1,:,:]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_se_resnext50(num_classes=28, **kwargs):
    return get_senet('se_resnext50_32x4d', num_classes=num_classes, **kwargs)


def get_model(config):
    print('model name:', config.model.name)
    f = globals().get('get_' + config.model.name)
    if config.model.params is None:
        return f()
    else:
        return f(**config.model.params)
    
    
# from torchsummary import summary
# model.cuda()
# summary(model, (4, 512, 512))