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

    def update(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backrpop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-eta/len(mini_batch)*nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-eta/len(mini_batch)*nb for b,nb in zip(self.biases,nabla_b)]

    def sgd(self,training_data,epochs,k,eta,test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[d:d+k] for d in range(0,n,k)]
            for mini_batch in mini_batches:
                self.update(mini_batch,eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)





tr_data, tst_data = load_data()
net = Network([784,100,10])

net.sgd(tr_data,30,10,0.01,tst_data)
#maximum with lambda 0.5 with mini-batch gradient descent - 86.7
# with lambda 0.08 maximum was 88.45
# with momentum when i set lambda to 0.8 i only got 1000 every time
# with momentum when I set lambda to 0.01 it got to 87.24

#batch size 20 lambda 0.01 / 86
'''
Epoch 0: 6042 / 10000
Epoch 1: 7314 / 10000
Epoch 2: 7766 / 10000
Epoch 3: 7992 / 10000
Epoch 4: 8133 / 10000
Epoch 5: 8207 / 10000
Epoch 6: 8282 / 10000
Epoch 7: 8319 / 10000
Epoch 8: 8366 / 10000
Epoch 9: 8392 / 10000
Epoch 10: 8441 / 10000
Epoch 11: 8436 / 10000
Epoch 12: 8443 / 10000
Epoch 13: 8475 / 10000
Epoch 14: 8482 / 10000
Epoch 15: 8516 / 10000
Epoch 16: 8528 / 10000
Epoch 17: 8551 / 10000
Epoch 18: 8548 / 10000
Epoch 19: 8553 / 10000
Epoch 20: 8571 / 10000
Epoch 21: 8557 / 10000
Epoch 22: 8585 / 10000
Epoch 23: 8601 / 10000
Epoch 24: 8590 / 10000
Epoch 25: 8600 / 10000
Epoch 26: 8617 / 10000
Epoch 27: 8631 / 10000
Epoch 28: 8638 / 10000
Epoch 29: 8626 / 10000
'''

#batch size 20 lambda 0.01 / 86
'''
Epoch 0: 5991 / 10000
Epoch 1: 7447 / 10000
Epoch 2: 7779 / 10000
Epoch 3: 8015 / 10000
Epoch 4: 8113 / 10000
Epoch 5: 8227 / 10000
Epoch 6: 8272 / 10000
Epoch 7: 8326 / 10000
Epoch 8: 8359 / 10000
Epoch 9: 8392 / 10000
Epoch 10: 8414 / 10000
Epoch 11: 8455 / 10000
Epoch 12: 8444 / 10000
Epoch 13: 8474 / 10000
Epoch 14: 8493 / 10000
Epoch 15: 8467 / 10000
Epoch 16: 8522 / 10000
Epoch 17: 8546 / 10000
Epoch 18: 8531 / 10000
Epoch 19: 8570 / 10000
Epoch 20: 8573 / 10000
Epoch 21: 8591 / 10000
Epoch 22: 8567 / 10000
Epoch 23: 8583 / 10000
Epoch 24: 8601 / 10000
Epoch 25: 8596 / 10000
Epoch 26: 8617 / 10000
Epoch 27: 8622 / 10000
Epoch 28: 8606 / 10000
Epoch 29: 8644 / 10000
'''