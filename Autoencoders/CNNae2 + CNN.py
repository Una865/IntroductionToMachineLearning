import keras
import numpy as np
import os
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import gzip
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D,UpSampling2D,Input
)

from keras.datasets import fashion_mnist

from keras.models import Model
from sklearn.model_selection import train_test_split

def encoder(input_img):

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(32,kernel_size=3,strides=2,padding='same',activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv3 = MaxPooling2D((2,2))(conv2)
    return conv3

def decoder(conv3):
    #decoder
    conv4 = UpSampling2D((2,2))(conv3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv6 = UpSampling2D((2,2))(conv5)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv6)
    return decoded

def fc(encode):
    flat = Flatten()(encode)
    den1 = Dense(128,activation = 'relu')(flat)
    out = Dense(10,activation = 'sigmoid')(den1)
    return out


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train,y_train,x_test,y_test

epochs = 100
batch_size = 32
inChannel = 3
x, y = 32, 32
input_img = Input(shape = (x, y, inChannel))
num_classes = 10

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam',metrics = ['accuracy'])
autoencoder.build = True
autoencoder.load_weights('autoencoder1.h5')
encode = encoder(input_img)

full = Model(input_img,fc(encode))

for l1,l2 in zip(full.layers[:7],autoencoder.layers[0:7]):
    l1.set_weights(l2.get_weights())

'''print(autoencoder.get_weights()[0][1])
print("Stop")
print(full.get_weights()[0][1])'''

for layer in full.layers[0:7]:
    layer.trainable = False

full.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


x_train,y_train,x_test,y_test = load_data()
loss,acc  = autoencoder.evaluate(x_test,x_test,verbose = 1)
print(acc)
full.fit(x_train,y_train,epochs = epochs,batch_size = 32,validation_data=(x_test,y_test),verbose=1)



'''after 44 epochs:
  accuracy on validation data was 57%'''
