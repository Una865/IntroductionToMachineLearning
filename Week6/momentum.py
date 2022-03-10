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

    def update(self,mini_batch,eta,ghama):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backrpop(x,y)
            nabla_b = [ghama*nb + eta*dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [ghama*nw+  eta*dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weights = [w-nw for w,nw in zip(self.weights,nabla_w)]
        self.biases = [b-nb for b,nb in zip(self.biases,nabla_b)]

    def sgd(self,training_data,epochs,k,eta,test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[d:d+k] for d in range(0,n,k)]
            for mini_batch in mini_batches:
                self.update(mini_batch,eta,0.9)

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
#batch size 10
# with momentum when i set lambda to 0.8 i only got 1000 every time
# with momentum when I set lambda to 0.01 it got to 87.24
# with 0.08 max was 87.03
# with 0.05 max was 87.28

#batch size 10 lambda 0.01 / 88
'''
Epoch 0: 8121 / 10000
Epoch 1: 8366 / 10000
Epoch 2: 8524 / 10000
Epoch 3: 8526 / 10000
Epoch 4: 8559 / 10000
Epoch 5: 8603 / 10000
Epoch 6: 8620 / 10000
Epoch 7: 8621 / 10000
Epoch 8: 8627 / 10000
Epoch 9: 8703 / 10000
Epoch 10: 8601 / 10000
Epoch 11: 8727 / 10000
Epoch 12: 8616 / 10000
Epoch 13: 8769 / 10000
Epoch 14: 8732 / 10000
Epoch 15: 8736 / 10000
Epoch 16: 8774 / 10000
Epoch 17: 8718 / 10000
Epoch 18: 8808 / 10000
Epoch 19: 8795 / 10000
Epoch 20: 8785 / 10000
Epoch 21: 8784 / 10000
Epoch 22: 8808 / 10000
Epoch 23: 8695 / 10000
Epoch 24: 8777 / 10000
Epoch 25: 8837 / 10000
Epoch 26: 8804 / 10000
Epoch 27: 8787 / 10000
Epoch 28: 8806 / 10000
Epoch 29: 8812 / 10000
'''


#batch size 20 lambda 0.05 / 87-70

#batch size 20 lambda 0.01 / 88
'''
Epoch 0: 7992 / 10000
Epoch 1: 8289 / 10000
Epoch 2: 8351 / 10000
Epoch 3: 8388 / 10000
Epoch 4: 8474 / 10000
Epoch 5: 8556 / 10000
Epoch 6: 8626 / 10000
Epoch 7: 8577 / 10000
Epoch 8: 8587 / 10000
Epoch 9: 8647 / 10000
Epoch 10: 8654 / 10000
Epoch 11: 8675 / 10000
Epoch 12: 8702 / 10000
Epoch 13: 8666 / 10000
Epoch 14: 8635 / 10000
Epoch 15: 8722 / 10000
Epoch 16: 8657 / 10000
Epoch 17: 8767 / 10000
Epoch 18: 8705 / 10000
Epoch 19: 8760 / 10000
Epoch 20: 8720 / 10000
Epoch 21: 8691 / 10000
Epoch 22: 8732 / 10000
Epoch 23: 8758 / 10000
Epoch 24: 8790 / 10000
Epoch 25: 8764 / 10000
Epoch 26: 8798 / 10000
Epoch 27: 8756 / 10000
Epoch 28: 8806 / 10000
Epoch 29: 8807 / 10000

'''