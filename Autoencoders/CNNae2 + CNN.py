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



'''
Epoch 1/100
1563/1563 [==============================] - 11s 7ms/step - loss: 1.4377 - accuracy: 0.4902 - val_loss: 1.3005 - val_accuracy: 0.5394
Epoch 2/100
1563/1563 [==============================] - 10s 7ms/step - loss: 1.2336 - accuracy: 0.5667 - val_loss: 1.2175 - val_accuracy: 0.5715
Epoch 3/100
1563/1563 [==============================] - 10s 7ms/step - loss: 1.1589 - accuracy: 0.5907 - val_loss: 1.2031 - val_accuracy: 0.5765
Epoch 4/100
1563/1563 [==============================] - 12s 8ms/step - loss: 1.1063 - accuracy: 0.6102 - val_loss: 1.2226 - val_accuracy: 0.5740
Epoch 5/100
1563/1563 [==============================] - 13s 8ms/step - loss: 1.0724 - accuracy: 0.6175 - val_loss: 1.1898 - val_accuracy: 0.5784
Epoch 6/100
1563/1563 [==============================] - 11s 7ms/step - loss: 1.0362 - accuracy: 0.6343 - val_loss: 1.2064 - val_accuracy: 0.5723
Epoch 7/100
1563/1563 [==============================] - 11s 7ms/step - loss: 1.0177 - accuracy: 0.6382 - val_loss: 1.1930 - val_accuracy: 0.5890
Epoch 8/100
1563/1563 [==============================] - 12s 7ms/step - loss: 0.9915 - accuracy: 0.6476 - val_loss: 1.1726 - val_accuracy: 0.5945
Epoch 9/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.9690 - accuracy: 0.6560 - val_loss: 1.1951 - val_accuracy: 0.5842
Epoch 10/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.9501 - accuracy: 0.6632 - val_loss: 1.1651 - val_accuracy: 0.5971
Epoch 11/100
1563/1563 [==============================] - 12s 8ms/step - loss: 0.9326 - accuracy: 0.6665 - val_loss: 1.1778 - val_accuracy: 0.5970
Epoch 12/100
1563/1563 [==============================] - 12s 8ms/step - loss: 0.9150 - accuracy: 0.6749 - val_loss: 1.2417 - val_accuracy: 0.5852
Epoch 13/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8981 - accuracy: 0.6787 - val_loss: 1.2010 - val_accuracy: 0.5939
Epoch 14/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8864 - accuracy: 0.6834 - val_loss: 1.1987 - val_accuracy: 0.5931
Epoch 15/100
1563/1563 [==============================] - 12s 8ms/step - loss: 0.8741 - accuracy: 0.6878 - val_loss: 1.1906 - val_accuracy: 0.5976
Epoch 16/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8596 - accuracy: 0.6932 - val_loss: 1.2277 - val_accuracy: 0.5928
Epoch 17/100
1563/1563 [==============================] - 12s 8ms/step - loss: 0.8490 - accuracy: 0.6970 - val_loss: 1.2678 - val_accuracy: 0.5816
Epoch 18/100
1563/1563 [==============================] - 12s 8ms/step - loss: 0.8367 - accuracy: 0.7013 - val_loss: 1.2446 - val_accuracy: 0.5904
Epoch 19/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8272 - accuracy: 0.7043 - val_loss: 1.2486 - val_accuracy: 0.5945
Epoch 20/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8132 - accuracy: 0.7076 - val_loss: 1.2742 - val_accuracy: 0.5911
Epoch 21/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.8033 - accuracy: 0.7125 - val_loss: 1.2468 - val_accuracy: 0.5936
Epoch 22/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7940 - accuracy: 0.7147 - val_loss: 1.2670 - val_accuracy: 0.5959
Epoch 23/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7844 - accuracy: 0.7198 - val_loss: 1.2887 - val_accuracy: 0.5834
Epoch 24/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7730 - accuracy: 0.7234 - val_loss: 1.2975 - val_accuracy: 0.5898
Epoch 25/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7649 - accuracy: 0.7264 - val_loss: 1.2992 - val_accuracy: 0.5892
Epoch 26/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.7535 - accuracy: 0.7298 - val_loss: 1.3540 - val_accuracy: 0.5853
Epoch 27/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7494 - accuracy: 0.7311 - val_loss: 1.3429 - val_accuracy: 0.5872
Epoch 28/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7390 - accuracy: 0.7335 - val_loss: 1.3303 - val_accuracy: 0.5859
Epoch 29/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.7350 - accuracy: 0.7351 - val_loss: 1.3659 - val_accuracy: 0.5808
Epoch 30/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.7212 - accuracy: 0.7423 - val_loss: 1.3432 - val_accuracy: 0.5880
Epoch 31/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.7157 - accuracy: 0.7426 - val_loss: 1.3696 - val_accuracy: 0.5819
Epoch 32/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.7124 - accuracy: 0.7432 - val_loss: 1.3796 - val_accuracy: 0.5840
Epoch 33/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.7028 - accuracy: 0.7478 - val_loss: 1.3968 - val_accuracy: 0.5835
Epoch 34/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6983 - accuracy: 0.7495 - val_loss: 1.3941 - val_accuracy: 0.5843
Epoch 35/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6903 - accuracy: 0.7510 - val_loss: 1.4055 - val_accuracy: 0.5829
Epoch 36/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6827 - accuracy: 0.7551 - val_loss: 1.4161 - val_accuracy: 0.5832
Epoch 37/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6771 - accuracy: 0.7560 - val_loss: 1.4221 - val_accuracy: 0.5807
Epoch 38/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6699 - accuracy: 0.7571 - val_loss: 1.4690 - val_accuracy: 0.5776
Epoch 39/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6658 - accuracy: 0.7594 - val_loss: 1.4950 - val_accuracy: 0.5808
Epoch 40/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6579 - accuracy: 0.7603 - val_loss: 1.5091 - val_accuracy: 0.5711
Epoch 41/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6503 - accuracy: 0.7649 - val_loss: 1.5109 - val_accuracy: 0.5692
Epoch 42/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.6456 - accuracy: 0.7667 - val_loss: 1.5141 - val_accuracy: 0.5800
Epoch 43/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6403 - accuracy: 0.7691 - val_loss: 1.5106 - val_accuracy: 0.5852
Epoch 44/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6333 - accuracy: 0.7707 - val_loss: 1.5249 - val_accuracy: 0.5704
Epoch 45/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6331 - accuracy: 0.7702 - val_loss: 1.5256 - val_accuracy: 0.5747
Epoch 46/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6270 - accuracy: 0.7718 - val_loss: 1.5896 - val_accuracy: 0.5611
Epoch 47/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6230 - accuracy: 0.7754 - val_loss: 1.5804 - val_accuracy: 0.5681
Epoch 48/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6165 - accuracy: 0.7762 - val_loss: 1.5581 - val_accuracy: 0.5694
Epoch 49/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6113 - accuracy: 0.7805 - val_loss: 1.5951 - val_accuracy: 0.5726
Epoch 50/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6043 - accuracy: 0.7824 - val_loss: 1.6077 - val_accuracy: 0.5678
Epoch 51/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.6022 - accuracy: 0.7810 - val_loss: 1.6565 - val_accuracy: 0.5586
Epoch 52/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.5970 - accuracy: 0.7842 - val_loss: 1.6338 - val_accuracy: 0.5729
Epoch 53/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.5917 - accuracy: 0.7868 - val_loss: 1.6459 - val_accuracy: 0.5732
Epoch 54/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.5872 - accuracy: 0.7889 - val_loss: 1.6503 - val_accuracy: 0.5712
Epoch 55/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.5852 - accuracy: 0.7878 - val_loss: 1.7359 - val_accuracy: 0.5596
Epoch 56/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5730 - accuracy: 0.7917 - val_loss: 1.6692 - val_accuracy: 0.5707
Epoch 57/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5755 - accuracy: 0.7911 - val_loss: 1.7387 - val_accuracy: 0.5642
Epoch 58/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5681 - accuracy: 0.7944 - val_loss: 1.7244 - val_accuracy: 0.5712
Epoch 59/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5645 - accuracy: 0.7960 - val_loss: 1.7682 - val_accuracy: 0.5636
Epoch 60/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5659 - accuracy: 0.7945 - val_loss: 1.7311 - val_accuracy: 0.5643
Epoch 61/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5545 - accuracy: 0.7980 - val_loss: 1.7735 - val_accuracy: 0.5664
Epoch 62/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5492 - accuracy: 0.7998 - val_loss: 1.8183 - val_accuracy: 0.5598
Epoch 63/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5472 - accuracy: 0.8011 - val_loss: 1.8047 - val_accuracy: 0.5671
Epoch 64/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5495 - accuracy: 0.7996 - val_loss: 1.8023 - val_accuracy: 0.5609
Epoch 65/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5389 - accuracy: 0.8037 - val_loss: 1.8150 - val_accuracy: 0.5667
Epoch 66/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5362 - accuracy: 0.8042 - val_loss: 1.8140 - val_accuracy: 0.5684
Epoch 67/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5346 - accuracy: 0.8054 - val_loss: 1.8005 - val_accuracy: 0.5698
Epoch 68/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5348 - accuracy: 0.8075 - val_loss: 1.8035 - val_accuracy: 0.5721
Epoch 69/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5213 - accuracy: 0.8107 - val_loss: 1.8734 - val_accuracy: 0.5659
Epoch 70/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5238 - accuracy: 0.8078 - val_loss: 1.8903 - val_accuracy: 0.5522
Epoch 71/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5176 - accuracy: 0.8124 - val_loss: 1.8919 - val_accuracy: 0.5635
Epoch 72/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5170 - accuracy: 0.8110 - val_loss: 1.8649 - val_accuracy: 0.5630
Epoch 73/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5179 - accuracy: 0.8117 - val_loss: 1.9222 - val_accuracy: 0.5561
Epoch 74/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5058 - accuracy: 0.8157 - val_loss: 1.9279 - val_accuracy: 0.5612
Epoch 75/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.5086 - accuracy: 0.8149 - val_loss: 1.9693 - val_accuracy: 0.5556
Epoch 76/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4964 - accuracy: 0.8180 - val_loss: 1.9532 - val_accuracy: 0.5567
Epoch 77/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.5020 - accuracy: 0.8173 - val_loss: 1.9747 - val_accuracy: 0.5612
Epoch 78/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.4991 - accuracy: 0.8168 - val_loss: 2.0114 - val_accuracy: 0.5589
Epoch 79/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4936 - accuracy: 0.8205 - val_loss: 2.0438 - val_accuracy: 0.5547
Epoch 80/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4932 - accuracy: 0.8198 - val_loss: 1.9915 - val_accuracy: 0.5603
Epoch 81/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4869 - accuracy: 0.8247 - val_loss: 2.0622 - val_accuracy: 0.5592
Epoch 82/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4858 - accuracy: 0.8219 - val_loss: 2.0793 - val_accuracy: 0.5512
Epoch 83/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4839 - accuracy: 0.8231 - val_loss: 2.0085 - val_accuracy: 0.5647
Epoch 84/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4808 - accuracy: 0.8243 - val_loss: 2.0397 - val_accuracy: 0.5633
Epoch 85/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4752 - accuracy: 0.8266 - val_loss: 2.0890 - val_accuracy: 0.5558
Epoch 86/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4766 - accuracy: 0.8262 - val_loss: 2.1003 - val_accuracy: 0.5610
Epoch 87/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4714 - accuracy: 0.8272 - val_loss: 2.1354 - val_accuracy: 0.5493
Epoch 88/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4661 - accuracy: 0.8312 - val_loss: 2.1543 - val_accuracy: 0.5479
Epoch 89/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4614 - accuracy: 0.8321 - val_loss: 2.1459 - val_accuracy: 0.5573
Epoch 90/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4642 - accuracy: 0.8300 - val_loss: 2.1300 - val_accuracy: 0.5528
Epoch 91/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4595 - accuracy: 0.8306 - val_loss: 2.1818 - val_accuracy: 0.5609
Epoch 92/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4570 - accuracy: 0.8331 - val_loss: 2.2205 - val_accuracy: 0.5586
Epoch 93/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4564 - accuracy: 0.8337 - val_loss: 2.1865 - val_accuracy: 0.5603
Epoch 94/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4531 - accuracy: 0.8352 - val_loss: 2.2490 - val_accuracy: 0.5568
Epoch 95/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.4487 - accuracy: 0.8371 - val_loss: 2.1841 - val_accuracy: 0.5562
Epoch 96/100
1563/1563 [==============================] - 11s 7ms/step - loss: 0.4462 - accuracy: 0.8383 - val_loss: 2.2601 - val_accuracy: 0.5589
Epoch 97/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4429 - accuracy: 0.8376 - val_loss: 2.2744 - val_accuracy: 0.5604
Epoch 98/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4453 - accuracy: 0.8375 - val_loss: 2.2576 - val_accuracy: 0.5501
Epoch 99/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4394 - accuracy: 0.8408 - val_loss: 2.3705 - val_accuracy: 0.5414
Epoch 100/100
1563/1563 [==============================] - 10s 7ms/step - loss: 0.4424 - accuracy: 0.8365 - val_loss: 2.2879 - val_accuracy: 0.5522
'''
