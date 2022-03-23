import tensorflow as tf
from tensorflow import keras
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import (
    Dense, Dropout, Embedding, Conv1D, MaxPool1D, GlobalMaxPool1D,Flatten
)
from keras.losses import BinaryCrossentropy
from keras.preprocessing import sequence
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(7)

max_words = 7000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)
top_words = 450
X_train = sequence.pad_sequences(X_train, maxlen=top_words)
X_test = sequence.pad_sequences(X_test, maxlen=top_words)

model = Sequential()
model.add(Embedding(max_words,64,input_length = top_words))
model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_1 (Embedding)     (None, 450, 128)          896000    
                                                                 
 conv1d_1 (Conv1D)           (None, 450, 128)          49280     
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 225, 128)         0         
 1D)                                                             
                                                                 
 flatten (Flatten)           (None, 28800)             0         
                                                                 
 dense (Dense)               (None, 256)               7373056   
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 8,318,593
Trainable params: 8,318,593
Non-trainable params: 0
_________________________________________________________________
'''

model.fit(X_train, y_train,validation_data = (X_test,y_test),epochs = 32,batch_size = 64,verbose = 1)
'''
Epoch 1/10
391/391 [==============================] - 59s 151ms/step - loss: 0.0113 - accuracy: 0.9965 - val_loss: 0.7800 - val_accuracy: 0.8664
Epoch 2/10
391/391 [==============================] - 59s 150ms/step - loss: 0.0023 - accuracy: 0.9993 - val_loss: 0.8913 - val_accuracy: 0.8710
Epoch 3/10
391/391 [==============================] - 59s 151ms/step - loss: 4.6920e-04 - accuracy: 0.9999 - val_loss: 0.9448 - val_accuracy: 0.8724
Epoch 4/10
391/391 [==============================] - 59s 151ms/step - loss: 8.6431e-05 - accuracy: 1.0000 - val_loss: 0.9632 - val_accuracy: 0.8739
Epoch 5/10
391/391 [==============================] - 59s 151ms/step - loss: 3.4350e-05 - accuracy: 1.0000 - val_loss: 0.9804 - val_accuracy: 0.8742
Epoch 6/10
391/391 [==============================] - 58s 149ms/step - loss: 2.3419e-05 - accuracy: 1.0000 - val_loss: 1.0069 - val_accuracy: 0.8744
Epoch 7/10
391/391 [==============================] - 58s 149ms/step - loss: 1.4717e-05 - accuracy: 1.0000 - val_loss: 1.0445 - val_accuracy: 0.8751
Epoch 8/10
391/391 [==============================] - 59s 150ms/step - loss: 8.8624e-06 - accuracy: 1.0000 - val_loss: 1.0821 - val_accuracy: 0.8752
Epoch 9/10
391/391 [==============================] - 59s 150ms/step - loss: 5.6888e-06 - accuracy: 1.0000 - val_loss: 1.1164 - val_accuracy: 0.8752
Epoch 10/10
391/391 [==============================] - 58s 149ms/step - loss: 3.8441e-06 - accuracy: 1.0000 - val_loss: 1.1492 - val_accuracy: 0.8754
<keras.callbacks.History at 0x7f3e36b964d0>
'''
