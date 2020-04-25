
1. !git clone https://github.com/iamfaith/ml_fw.git

2. %cd ml_fw

3. !pip install wandb

4. code

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

