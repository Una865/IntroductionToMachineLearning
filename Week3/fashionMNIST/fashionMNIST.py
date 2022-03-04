import numpy as np

from keras.datasets import fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
trainy = np.reshape(trainy,(60000,1))

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def positive(x, th, th0):
    return np.sign(th.T@x + th0)

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
    return xval_learning_alg(lambda data, labels: averaged_perceptron(data, labels, {"T": 50}), data, labels, 10)


def raw_mnist_features(x):

    n_samples, m, n = x.shape
    x = np.reshape(x, (n_samples,m*n))
    return x.T

def compare(n1,n2,data):

    d1 = data[n1]['images']
    d2 = data[n2]['images']

    train_data = np.vstack((d1[:900], d2[:900]))
    test_data = np.vstack((d1[900:], d2[900:]))


    y1 = np.repeat(-1, len(d1[:900])).reshape(1, -1)
    y2 = np.repeat(1, len(d2[:900])).reshape(1, -1)
    train_labels = np.vstack((y1.T, y2.T)).T

    y1 = np.repeat(-1, len(d1[900:])).reshape(1, -1)
    y2 = np.repeat(1, len(d2[900:])).reshape(1, -1)
    test_labels = np.vstack((y1.T, y2.T)).T

    print(eval_classifier(averaged_perceptron,raw_mnist_features(train_data),train_labels,raw_mnist_features(test_data),test_labels))


def load_data(labels):

    data = {}

    for label in labels:

        imgs = []
        y = np.array([[label] * 1000])
        i = 0
        cnt = 0
        while cnt < 1000:
            if trainy[i] == label:
                imgs.append(trainX[i]/255)
                cnt+=1
            i+=1
        y = np.array([[label] * 1000])
        data[label] = {
            'images': imgs,
            'labels': y
        }


    return data



data = load_data(range(10))
compare(1,5,data)







