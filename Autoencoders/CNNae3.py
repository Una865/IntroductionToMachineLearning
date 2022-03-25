import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import numpy as np
import gzip
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Conv2DTranspose,Reshape,BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D,UpSampling2D,Input
)

from keras.datasets import fashion_mnist
from PIL import Image as im
from keras.models import Model
from sklearn.model_selection import train_test_split




def encoder(input_img):

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = MaxPooling2D((2,2))(conv2)
    conv4 = Flatten()(conv3)
    conv4 = Dense(100)(conv4)
    return conv4

def decoder(encode):
    #decoder
    conv5 = Dense(4096)(encode)
    conv6 = Reshape((8,8,64))(conv5)
    conv7 = UpSampling2D()(conv6)
    conv8 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(conv7)
    conv9 = BatchNormalization()(conv8)
    conv9 =UpSampling2D()(conv9)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    decoded = Conv2D(3,(3,3),activation='sigmoid',padding='same')(conv10)
    return decoded

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = load_data()
epochs = 10
batch_size = 64
inChannel = 3
x, y = 32, 32
input_img = Input(shape = (x, y,inChannel))
num_classes = 10

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam',metrics = ['mse'])
autoencoder.summary()

train_X,valid_X,train_ground,valid_ground = train_test_split(x_train,
                                                             x_train,
                                                             test_size=0.2,
                                                             random_state=13)

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))
autoencoder.save_weights('autoencoder2.h5')




