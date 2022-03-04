import csv

import numpy as np
import pdb


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

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)

def eval_classifier(learner, data_train, labels_train, data_test, labels_test,params = {}):
    th, th0 = learner(data_train, labels_train,params)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]

def xval_learning_alg(learner, data, labels, k,params ={}):
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
                                              data_test, labels_test,params)
    return score_sum/k

def load_auto_data(path_data):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    return [(v-std[0])/std[1]]

def raw(x):
    return [x]

def one_hot(v, entries):
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features
def auto_data_and_labels(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    print('avg and std', std)
    print('entries in one_hot field', entries)
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    return data_labels[:, 1:].T, data_labels[:, 0:1].T

auto_data_all = load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', one_hot),
            ('displacement', standard),
            ('horsepower', standard),
            ('weight', standard),
            ('acceleration', standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)