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
model.add(Embedding(max_words,128,input_length = top_words))
model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation = 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

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
