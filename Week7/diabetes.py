import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import model_selection
from sklearn import metrics



data = pd.read_csv('diabetes.csv')
Y = data['Outcome']
X = data.drop(columns='Outcome')


x_train,x_test,y_train, y_test = train_test_split(X, Y,test_size = 0.2,shuffle=True,stratify = Y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)
print(x_test.shape)
model = Sequential()
model.add(Dense(units = 32,input_dim=8,activation='relu'))
model.add(Dense(units = 16,activation='relu'))
model.add(Dense(units = 1,activation='softmax'))

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs = 50, batch_size = 32)

test_scores = model.evaluate(x_test,y_test,batch_size=32)
print('Skup za testiranje {0}: {1}'.format(model.metrics_names[1], test_scores[1]))


'''
*************************************************************
model = Sequential()
model.add(Dense(units = 15,input_dim=8,activation='relu'))
model.add(Dense(units = 8,activation='relu'))
model.add(Dense(units = 1,activation='sigmoid'))

accuracy: 0.7337662577629089
*************************************************************

*************************************************************
model = Sequential()
model.add(Dense(units = 15,input_dim=8,activation='relu'))
model.add(Dense(units = 8,activation='relu'))
model.add(Dense(units = 1,activation='softmax'))

accuracy: 0.350649356842041
*************************************************************
'''

