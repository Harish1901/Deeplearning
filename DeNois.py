# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:56:39 2024

@author: haris
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


from keras import layers
from keras.datasets import mnist
from keras.models import Model


data_path = r'C:\Users\haris\Downloads\Denoise\label'
test_path = r'C:\Users\haris\Downloads\Denoise\roadway'
ori_path =  r'C:\Users\haris\Downloads\Denoise\rainyimages'
data_dir_list = os.listdir(data_path)
test_dir_list = os.listdir(test_path)
orig_dir_list = os.listdir(ori_path)
img_data_list = []
test_data_list = []
ori_data_list = []

def preprocess(array, path, data_list):
    """Normalizes the supplied array and reshapes it."""
    for img in array:
        inp_img = cv2.imread(path+'/'+img)
        inp_img = cv2.cvtColor(inp_img,cv2.COLOR_BGR2GRAY)
        inp_img_resize = cv2.resize(inp_img,(300,300))
        data_list.append(inp_img_resize)
    img_data = np.array(data_list)
    img_data = img_data.astype('float32') 
    array = img_data.reshape(-1,300, 300, 1)/255
    return array

def noise(array):
    """Adds random noise to each image in the supplied array."""
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """Displays ten random images from each array."""
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    
    
    
# Normalize and reshape the data
train_data = preprocess(data_dir_list, data_path, img_data_list)
test_data = preprocess(test_dir_list, test_path, test_data_list)
orig_test_data =  preprocess(orig_dir_list, ori_path, ori_data_list)

# Create a copy of the data with added noise
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)
noisy_origtest_data = noise(orig_test_data)

# Display the train data and a version of it with added noise
display(train_data, noisy_train_data)
display(test_data, noisy_test_data)





input = layers.Input(shape=(300, 300, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

#Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()


# autoencoder.fit(
#     x=train_data,
#     y=train_data,
#     epochs=10,
#     batch_size=20,
#     shuffle=True,
#     validation_data=(test_data, test_data),
# )

# predictions = autoencoder.predict(test_data)
# display(test_data, predictions)


autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=10,
    batch_size=20,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)

predictions = autoencoder.predict(noisy_origtest_data)
display(noisy_origtest_data, predictions)