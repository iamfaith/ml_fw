


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