import numpy as np

def norm(col_v):
    #norm of the column vector
    sum = np.sum(np.square(col_v))
    return np.sqrt(sum)

def signed_dist(x,th,th0):
    #x - column vector dX1
    #th - theta vector dX1
    #th0 - theta0 vector 1X1

    return (np.dot(th.T,x)+th0)/norm(th)

def positive(x,th,th0):
    # 1 indicates it is on a positive side of hyperplane
    # 0 indicates it is in hyperplane
    # -1 otherwise
    return np.sign(signed_dist(x,th,th0))

def score(data,labels,th,th0):
    # return a scalar indicating how many of data points the separator correctly classified
    return np.sum(np.array(labels == positive(data, th, th0)))

# perceptron algorithms
def perceptron(data,labels,params = {}):

    T = params.get('T',100)

    d,n = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))

    for t in range(T):
        for i in range(n):
            X = data[:,i]
            Y = labels[0,i]
            if Y*(np.dot(X,th)+th0) <=0:
                th[:,0] = th[:,0] + Y*X
                th0 = th0 + Y
    return(th,th0)

def averaged_perceptron(data,labels,params = {}):
    T = params.get('T',100)

    d, n = data.shape
    th = np.zeros((d, 1))
    th0 = np.zeros((1, 1))

    ths = np.zeros((d, 1))
    th0s = np.zeros((1, 1))

    for t in range(T):
        for i in range(n):
            X = data[:, i]
            Y = labels[0, i]
            if Y * (np.dot(X, th) + th0) <= 0:
                th[:, 0] = th[:, 0] + Y * X
                th0 = th0 + Y
            ths+=th
            th0s+=th0
    return (ths/(n*T), th0s/(n*T))

#end of perceprton algorithms

#evaluating algorithms

def eval_classfier(learner, data_train, labels_train, data_test, labels_test):
    # data - a dXn array of floats
    # labels - a 1Xn array of elements -1/+1
    # th - a dX1 array of floats
    # th0 - a single scalar

    # return the percentage correct on a new testing set
    th,th0 = learner(data_train,labels_train)
    d, n = labels_test.shape
    scr = score(data_test,labels_test,th,th0)

    return scr/n

def eval_learning_alg(learner, data_gen, n_train, n_test,it):
    # it - number of times to evaluate learning classifier
    # every time it generates dat afor training and testing
    score =  0
    for i in range(it):

        all_data, all_labels = data_gen(n_train+n_test)

        data_train = all_data[:,:n_train]
        data_testing = all_data[:,n_train:]

        labels_train = all_labels[:,:n_train]
        labels_testing = all_labels[:,n_train:]

        scrCurr = eval_classfier(learner,data_train,labels_train,data_testing,labels_testing)
        score+=scrCurr

    return score/it

def xval_learning_alg(learner,data,labels,k):
    
    data_split = np.array_split(data,k,axis = 1)
    labels_split = np.array_split(labels,k,axis = 1)

    score = 0

    for i in range(k):
        data_train = np.concatenate(data_split[:i]+data_split[i+1:],axis = 1)
        data_test = np.array(data_split[i])

        labels_train = np.concatenate(labels_split[:i]+labels_split[i+1:],axis = 1)
        labels_test = np.array(labels_split[i])

        score+= eval_classfier(learner,data_train,labels_train,data_test,labels_test)

    return score/k
