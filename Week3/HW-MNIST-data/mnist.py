import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.image import imread

######################################################################
#helper functions

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def y(x, th, th0):
    # x is dimension d by n
    # th is dimension d by m
    # th0 is dimension 1 by m
    # return matrix of y values for each column of x and theta: dimension m x n
    return np.dot(np.transpose(th), x) + np.transpose(th0)

def length(d_by_m):
    return np.sum(d_by_m * d_by_m, axis = 0, keepdims = True)**0.5

# x is dimension d by n
# th is dimension d by m
# th0 is dimension 1 by m
# return matrix of signed dist for each column of x and theta: dimension m x n
def signed_dist(x, th, th0):
    return y(x, th, th0) / np.transpose(length(th))

#end of helper functions
######################################################################

######################################################################
# Perceptron code

# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
# Perceptron algorithm with offset.
# data is dimension d by n
# labels is dimension 1 by n
# T is a positive integer number of steps to run
def perceptron(data, labels, params = {}, hook = None):
    # if T not in params, default to 50
    T = params.get('T', 50)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
    return theta, theta_0

def averaged_perceptron(data, labels, params = {}, hook = None):
    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1)); theta_0 = np.zeros((1, 1))
    theta_sum = theta.copy()
    theta_0_sum = theta_0.copy()
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
                if hook: hook((theta, theta_0))
            theta_sum = theta_sum + theta
            theta_0_sum = theta_0_sum + theta_0
    theta_avg = theta_sum / (T*n)
    theta_0_avg = theta_0_sum / (T*n)
    if hook: hook((theta_avg, theta_0_avg))
    return theta_avg, theta_0_avg

#end of perceptron code
######################################################################

######################################################################
#evaluation codes

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    th, th0 = learner(data_train, labels_train)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]

def xval_learning_alg(learner, data, labels, k):
    _, n = data.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    data, labels = data[:,idx], labels[:,idx]

    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k


def get_classification_accuracy(data, labels):

    # data (d,n) array
    # labels (1,n) array
    return xval_learning_alg(lambda data, labels: perceptron(data, labels, {"T": 50}), data, labels, 10)

#end of evaluation codes
######################################################################

######################################################################
#loading data
def load_mnist_data(labels):

    # labels list of labels from {0, 1,...,9}
    # returns dict: label (int) -> [[image1], [image2], ...]

    data = {}

    for label in labels:
        images = load_mnist_single("mnist/mnist_train{}.png".format(label))
        y = np.array([[label] * len(images)])
        data[label] = {
            "images": images,
            "labels": y
        }

    return data


def load_mnist_single(path_data):

    #return list of images (first row of large picture)


    img = imread(path_data)  # 2156 x 2156 (m,n)
    m, n = img.shape

    side_len = 28  # standard mnist
    n_img = int(m / 28)

    imgs = []  # list of single images
    for i in range(n_img):
        start_ind = i*side_len
        end_ind = start_ind + side_len
        current_img = img[start_ind:end_ind, :side_len]  # 28 by 28

        current_img = current_img / 255 # normalization!!!

        imgs.append(current_img)

    return imgs

#end of loading functions
######################################################################

######################################################################
#feature transformation


def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n_samples, m, n = x.shape
    x = np.reshape(x, (n_samples,m*n))
    return x.T


def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    n_samples, m, n = x.shape
    avg = np.zeros((n_samples, m,1))
    for i in range(n_samples):
        avg[i] = np.transpose([np.mean(x[i], axis=1)])
    avg = np.reshape(avg,(n_samples,m))
    return avg.T


def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n_samples, m, n = x.shape
    avg = np.zeros((n_samples, n,1))
    for i in range(n_samples):
        avg[i] = np.transpose([np.mean(x[i], axis=0)])
    avg = np.reshape(avg, (n_samples, n))
    return avg.T

def top_bottom_features_onne(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m,n = x.shape
    ind = (int)(m/2)
    top_half = x[:ind]
    bot_half = x[ind:]
    s1 = 0
    s2 = 0
    for i in range(ind):
        s1 += np.sum(cv(x[i,:]))
    for i in range(ind,m):
        s2+=np.sum(cv(x[i,:]))
    return np.array([[s1/(ind*n)],[s2/((m-ind)*n)]])


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    n_samples,m,n = x.shape
    arr = np.zeros((2,n_samples))
    for i in range(n_samples):
        curr = top_bottom_features_onne(x[i])
        arr[0][i] = curr[0]
        arr[1][i] = curr[1]
    return arr

#end of feature transformation
######################################################################


mnist_data_all = load_mnist_data(range(10))
print(mnist_data_all[0]['images'][0].shape)


d0 = mnist_data_all[8]["images"]
d1 = mnist_data_all[9]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T
acc = get_classification_accuracy(col_average_features(data), labels)
print(acc)

f = open("results-8vs9.txt", "a")
for i in range(4):
    if i ==1:
        print("When using raw data accuracy is: " , get_classification_accuracy(raw_mnist_features(data),labels),file=f)
    if i==2:
        print("When using col_average accuracy is: " , get_classification_accuracy(col_average_features(data), labels),
              file=f)
    if i==3:
        print("When using row_average accuracy is: ", get_classification_accuracy(row_average_features(data), labels),
              file=f)
    if i==4:
        print("When using top-bottom accuracy is: " , get_classification_accuracy(top_bottom_features(data), labels),
              file=f)

