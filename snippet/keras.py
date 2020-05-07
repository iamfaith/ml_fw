from keras.utils.vis_utils import plot_model
plot_model(model, to_file="model.png", show_shapes=True)


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import init,importlib
importlib.reload(init)
print(init.cfg)


from transform import transform
import transform as trans

importlib.reload(transform)
importlib.reload(init)
# print(trans, transform)
train_datagen = transform.get_transform(init.cfg, 'train')
test_datagen = transform.get_transform(init.cfg, 'test')
print(train_datagen, test_datagen)



from model import model_factory
model = model_factory.keras_InceptionV3(11)


from checkpoint import cp
from keras.optimizers import SGD
checkpointer = cp.save_checkpoint(init.cfg)
print(checkpointer)
train_param = init.cfg['train']
model.compile(optimizer=SGD(lr=train_param['learning_rate'], momentum=0.9, decay = 0.0, nesterov = True), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(train_datagen.flow(train_x, trans(train_y), batch_size=train_param['batch_size']), 
                    epochs=train_param['epochs'],
                    validation_data = test_datagen.flow(val_x, trans(val_y), batch_size=train_param['batch_size']),
                    callbacks = checkpointer,
                    verbose=1
                    )





def data_generator(data, batch_size): #样本生成器，节省内存
    while True:
        # print(data, batch_size)
        batch = np.random.choice(data, batch_size)
        x,y = [],[]
        for img in batch:
            # img_gray = Image.open(img).convert('L')
            # img_two = img_gray.point(lambda x: 255 if x > 129 else 0)

            gray = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
            # print(img, type(img_two))       
            one_channel = cv2.resize(gray, (width, height)) 
            x.append(np.array([one_channel, one_channel, one_channel]).transpose(1,2,0)  )
            y.append([keys.get(i) for i in img[-8:-4].lower()])
        x = preprocess_input(np.array(x).astype(float))
        y = np.array(y)
        tmp_y = [y[:,i] for i in range(4)]
        # print(x.shape, y.shape, len(tmp_y[0]))
        yield x, tmp_y
        
        
model.fit_generator(data_generator(train_samples, 100), steps_per_epoch=1000, epochs=10, 
                    validation_data=data_generator(test_samples, 100), validation_steps=100,
                    callbacks = checkpointer,
                    verbose=1
                    ) 