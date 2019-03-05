from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from time import time
from keras.callbacks import TensorBoard

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '/Users/Santhosh/Downloads/all/train1'
validation_data_dir = '/Users/Santhosh/Downloads/all/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 1
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
tensorboard= TensorBoard(log_dir='NULL'.format(time()))
hist=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,callbacks=[tensorboard],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# plotting results

train_loss=hist.history['loss']
train_accuracy=hist.history['acc']
validation_loss=hist.history['val_loss']
validation_accuracy=hist.history['val_acc']
xc=range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,validation_loss)
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title('train loss vs validation loss')
plt.grid(True)
plt.legend(['train','val'])
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_accuracy)
plt.plot(xc,validation_accuracy)
plt.xlabel('number of epochs')
plt.ylabel('Accuracy')
plt.title('train accuracy vs validation accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])
model.save_weights('first_try.h5')

