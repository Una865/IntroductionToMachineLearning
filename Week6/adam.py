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

    def update(self,mini_batch,eta,eps,B1,B2):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m_b = [np.zeros(b.shape) for b in self.biases]
        m_w = [np.zeros(w.shape) for w in self.weights]
        v_b = [np.zeros(b.shape) for b in self.biases]
        v_w = [np.zeros(w.shape) for w in self.weights]
        eta1 = eta
        last_w,last_b = [],[]
        cnt = 0
        for x,y in mini_batch:
            cnt+=1
            delta_nabla_b,delta_nabla_w = self.backrpop(x,y)
            m_b = [B1*mb + (1-B1)*dnb for mb,dnb in zip(m_b,delta_nabla_b)]
            m_b = [mb/(1-(B1**cnt)) for mb in m_b]

            v_b = [B2*vb+(1-B2)*(dnb**2)for vb,dnb in zip(v_b,delta_nabla_b)]
            v_b = [vb / (1 - (B2 ** cnt)) for vb in v_b]

            m_w = [B1*mw+(1-B1)*dnw for mw,dnw in zip(m_w,delta_nabla_w)]
            m_w = [mw / (1 - (B1 ** cnt)) for mw in m_w]
            v_w = [B2*vw + (1 - B2) * (dnw**2) for vw, dnw in zip(v_w, delta_nabla_w)]
            v_w = [vw / (1 - (B2 ** cnt)) for vw in v_w]

            eta1 =np.sqrt((1-(B2**cnt))/(1-(B1**cnt)))*eta1
            self.weights = [w - eta / np.sqrt(vw + eps) * mw for w, vw, mw in zip(self.weights, v_w, m_w)]
            self.biases = [b - eta / np.sqrt(vb + eps) * mb for b, vb, mb in zip(self.biases, v_b, m_b)]



        #self.weights = [w-eta1/np.sqrt(vw+eps)*mw for w,vw,mw in zip(self.weights,v_w,m_w)]
        #self.biases = [b-eta1/np.sqrt(vb+eps)*mb for b,vb,mb in zip(self.biases,v_b,m_b)]

    def sgd(self,training_data,epochs,k,eta,test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[d:d+k] for d in range(0,n,k)]
            for mini_batch in mini_batches:
                self.update(mini_batch,eta,1e-8,0.9,0.999)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def evaluate(self,test_data):
        test_results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)





tr_data, tst_data = load_data()
net = Network([784,100,10])

net.sgd(tr_data,30,10,0.001,tst_data)

'''
net.sgd(tr_data,30,10,0.001,tst_data)
Epoch 0: 1853 / 10000
Epoch 1: 1989 / 10000
Epoch 2: 3498 / 10000
Epoch 3: 4406 / 10000
Epoch 4: 4531 / 10000
Epoch 5: 4569 / 10000
Epoch 6: 4548 / 10000
Epoch 7: 4571 / 10000
Epoch 8: 4720 / 10000
Epoch 9: 5517 / 10000
Epoch 10: 5532 / 10000
Epoch 11: 5541 / 10000
Epoch 12: 5549 / 10000
Epoch 13: 5579 / 10000
Epoch 14: 5564 / 10000
Epoch 15: 5572 / 10000
Epoch 16: 5586 / 10000
Epoch 17: 5601 / 10000
Epoch 18: 5582 / 10000
Epoch 19: 5607 / 10000
Epoch 20: 5594 / 10000
Epoch 21: 5610 / 10000
Epoch 22: 5610 / 10000
Epoch 23: 5612 / 10000
Epoch 24: 5603 / 10000
Epoch 25: 5596 / 10000
Epoch 26: 5770 / 10000
Epoch 27: 6282 / 10000
Epoch 28: 6431 / 10000
Epoch 29: 6474 / 10000

'''
