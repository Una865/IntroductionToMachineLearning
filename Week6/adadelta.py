import random

import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from scipy.special import expit, logit

def load_data():

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    train_y = tf.keras.utils.to_categorical(y_train)
    test_y = tf.keras.utils.to_categorical(y_test)

    y_train = [np.reshape(y,(10,1)) for y in train_y]
    y_test = [np.reshape(y, (10, 1)) for y in test_y]

    train_x = [np.reshape(x,(784,1)) for x in X_train]
    test_x = [np.reshape(x,(784,1)) for x in X_test]

    x_train= [x/255.0 for x in train_x]
    x_test = [x / 255.0 for x in test_x]

    training_data = list(zip(x_train,y_train))
    test_data = list(zip(x_test,y_test))


    return training_data,test_data

def sigmoid(z):
    try:
     return 1.0/(1.0+np.exp(-z))
    except OverflowError as err:
        print(z.shape)
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):

        # initializing neural network
        # sizes is the list with number of neurons in i-th layer
        #initializing weights and biases using normal distribution with mean 0 and variance 1
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.normal(loc = 0,scale = 1/sizes[0],size = (y,1)) for y in sizes[1:]]
        self.weights = [np.random.normal(loc = 0,scale = 1/sizes[0],size = (y,x)) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,a)+b
            a = expit(z)

        return a
    def cost_derivative(self,output_activations,y):
        return output_activations-y
    def backrpop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = expit(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].T)

        for l in range(2,self.num_layers):
            z = zs[-l]
            sp = expit(z)*(1-expit(z))
            delta = np.dot(self.weights[-l+1].T,delta)*sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].T)

        return nabla_b,nabla_w

    def update(self,mini_batch,eta,ghama,eps):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        last_b = [np.zeros(b.shape) for b in self.biases]
        last_w = [np.zeros(w.shape) for w in self.weights]
        cnt = 0
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backrpop(x,y)
            last_b = delta_nabla_b
            last_w = delta_nabla_w
            nabla_b = [ghama*nb + (1-ghama)*(dnb**2) for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [ghama*nw+  (1-ghama)*(dnw**2) for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-eta/np.sqrt(nw+eps)*lnw for w,nw,lnw in zip(self.weights,nabla_w,last_w)]
        self.biases = [b-eta/np.sqrt(nb+eps)*lnb for b,nb,lnb in zip(self.biases,nabla_b,last_b)]

    def sgd(self,training_data,epochs,k,eta,test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[d:d+k] for d in range(0,n,k)]
            for mini_batch in mini_batches:
                self.update(mini_batch,eta,0.9,1e-6)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)





tr_data, tst_data = load_data()
net = Network([784,100,10])

net.sgd(tr_data,30,10,0.2,tst_data)
'''
net.sgd(tr_data,30,10,0.05,tst_data)
Epoch 0: 4647 / 10000
Epoch 1: 5222 / 10000
Epoch 2: 5984 / 10000
Epoch 3: 6498 / 10000
Epoch 4: 6529 / 10000
Epoch 5: 6692 / 10000
Epoch 6: 6293 / 10000
Epoch 7: 6645 / 10000
Epoch 8: 6606 / 10000
Epoch 9: 6692 / 10000
Epoch 10: 6870 / 10000
Epoch 11: 6809 / 10000
Epoch 12: 6897 / 10000
Epoch 13: 6952 / 10000
Epoch 14: 6964 / 10000
Epoch 15: 6813 / 10000
Epoch 16: 6454 / 10000
Epoch 17: 6618 / 10000
Epoch 18: 6997 / 10000
Epoch 19: 6811 / 10000
Epoch 20: 6785 / 10000
Epoch 21: 5978 / 10000
Epoch 22: 7075 / 10000
Epoch 23: 6988 / 10000
Epoch 24: 6905 / 10000
Epoch 25: 6906 / 10000
Epoch 26: 6922 / 10000
Epoch 27: 6663 / 10000
Epoch 28: 6821 / 10000
Epoch 29: 7148 / 10000

'''
'''
net.sgd(tr_data,30,10,0.1,tst_data)
Epoch 0: 4812 / 10000
Epoch 1: 6269 / 10000
Epoch 2: 5896 / 10000
Epoch 3: 5913 / 10000
Epoch 4: 6146 / 10000
Epoch 5: 5625 / 10000
Epoch 6: 6042 / 10000
Epoch 7: 6460 / 10000
Epoch 8: 5603 / 10000
Epoch 9: 6447 / 10000
Epoch 10: 6025 / 10000
Epoch 11: 5951 / 10000
Epoch 12: 6912 / 10000
Epoch 13: 6838 / 10000
Epoch 14: 6736 / 10000
Epoch 15: 6749 / 10000
Epoch 16: 6730 / 10000
Epoch 17: 5962 / 10000
Epoch 18: 6118 / 10000
Epoch 19: 6872 / 10000
Epoch 20: 6429 / 10000
Epoch 21: 6489 / 10000
Epoch 22: 6185 / 10000
Epoch 23: 6085 / 10000
Epoch 24: 6845 / 10000
Epoch 25: 6117 / 10000
Epoch 26: 6675 / 10000
Epoch 27: 5685 / 10000
Epoch 28: 6856 / 10000
Epoch 29: 6062 / 10000
'''

'''
net.sgd(tr_data,30,20,0.01,tst_data)
ghama ,  eps = 0.9,1e-6
Epoch 0: 6886 / 10000
Epoch 1: 6733 / 10000
Epoch 2: 7021 / 10000
Epoch 3: 7389 / 10000
Epoch 4: 7454 / 10000
Epoch 5: 7430 / 10000
Epoch 6: 7610 / 10000
Epoch 7: 7637 / 10000
Epoch 8: 7635 / 10000
Epoch 9: 7698 / 10000
Epoch 10: 7676 / 10000
Epoch 11: 7774 / 10000
Epoch 12: 7796 / 10000
Epoch 13: 7753 / 10000
Epoch 14: 7884 / 10000
Epoch 15: 7926 / 10000
Epoch 16: 7698 / 10000
Epoch 17: 7832 / 10000
Epoch 18: 7676 / 10000
Epoch 19: 7867 / 10000
Epoch 20: 7540 / 10000
Epoch 21: 8061 / 10000
Epoch 22: 7349 / 10000
Epoch 23: 7955 / 10000
Epoch 24: 7939 / 10000
Epoch 25: 7973 / 10000
Epoch 26: 7960 / 10000
Epoch 27: 7930 / 10000
Epoch 28: 7960 / 10000
Epoch 29: 7769 / 10000
'''

#net.sgd(tr_data,30,10,0.01,tst_data)
'''
Epoch 0: 6135 / 10000
Epoch 1: 6742 / 10000
Epoch 2: 7232 / 10000
Epoch 3: 7714 / 10000
Epoch 4: 7683 / 10000
Epoch 5: 7593 / 10000
Epoch 6: 7863 / 10000
Epoch 7: 7835 / 10000
Epoch 8: 7909 / 10000
Epoch 9: 7629 / 10000
Epoch 10: 7977 / 10000
Epoch 11: 7893 / 10000
Epoch 12: 7823 / 10000
Epoch 13: 7602 / 10000
Epoch 14: 7720 / 10000
Epoch 15: 7939 / 10000
Epoch 16: 7923 / 10000
Epoch 17: 7857 / 10000
Epoch 18: 7651 / 10000
Epoch 19: 7431 / 10000
Epoch 20: 7966 / 10000
Epoch 21: 7949 / 10000
Epoch 22: 7693 / 10000
Epoch 23: 7739 / 10000
Epoch 24: 7985 / 10000
Epoch 25: 7800 / 10000
Epoch 26: 7894 / 10000
Epoch 27: 7877 / 10000
Epoch 28: 7542 / 10000
Epoch 29: 8160 / 10000
'''