import csv
import pandas as pd
import numpy as np

def load_fish_data(path_data):

    numeric_fields = {'Species', 'Weight', 'Length1', 'Length2', 'Length3',
                      'Height', 'Width'}
    data = []
    with open(path_data,encoding="utf8") as f_data:
        for datum in csv.DictReader(f_data, delimiter=','):

            keys = list(datum.keys())
            for field in keys:
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

def raw_data(data):
    n = len(data)
    d = len(data[0])
    names = {
         'Bream':0,
         'Roach':1,
         'Whitefish':2,
         'Parkki':3,
         'Perch':4,
         'Pike':5,
         'Smelt':6
    }
    dt = np.array([]).reshape(0,d)

    for i in range(n):
        values = list(data[i].values())
        keys = list(data[i].keys())
        data[i][keys[0]] = names[data[i][keys[0]]]
        vals = np.array([list(data[i].values())])
        dt = np.vstack([dt,vals])



    np.random.shuffle(dt)
    return dt

def gen_data(data,n1,n2):
    n,d = data.shape

    train = np.array([]).reshape(0,d)
    for i in range(n):
        if data[i,0] == n1 or data[i,0] == n2:
            train = np.vstack([train,data[i,:]])


    train = train.T
    d, n = train.shape

    idx = int(0.9*n)

    X = train[1:,:idx]
    Y = train[:1,:idx]
    X_test = train[1:,idx:]
    Y_test = train[1:,idx:]

    return X,Y,X_test,Y_test


'''def gen_data(data):
    n, d = data.shape
    row_idx = np.array(range(n))
    col_idx = np.array(range(d))
    idx = int(0.9 * n)
    train_data = data[row_idx[:idx, None], col_idx[:]]
    test_data = data[row_idx[idx:, None], col_idx[:]]

    train_labels = labels[:idx]
    test_labels = labels[idx:]

    return train_data,train_labels,test_data,test_labels
'''
def d_hinge(v):
    return np.where(v >= 1, 0, -1)
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0))* y * x
def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0)) * y
def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis = 1, keepdims = True) + lam * 2 * th
def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis = 1, keepdims = True)
def svm_obj_grad(X, y, th, th0, lam):
    grad_th = d_svm_obj_th(X, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])
def hinge(v):
    return np.where(v >= 1, 0, 1 - v)

def hinge_loss(x, y, th, th0):
    return hinge(y * (np.dot(th.T, x) + th0))

def svm_obj(X, y, th, th0, lam):
    return np.mean(hinge_loss(X, y, th, th0)) + lam * np.linalg.norm(th) ** 2
def num_grad(f, delta=0.001):
    def df(x):
        n, d = x.shape
        gd = np.array([[]])
        for i in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += delta
            x2[i] -= delta
            v = np.array([[(f(x1) - f(x2)) / (2 * delta)]])
            if i == 0:
                gd = v.copy()
            else:
                gd = np.concatenate((gd, v), axis=0)
        return gd
        pass

    return df

def gd(f, df, x0, step_size_fn, max_iter):
    fs = [f(x0)]
    xs = [x0]
    for i in range(max_iter):
        x0 = (x0 - step_size_fn(i) * df(x0)).copy()
        fs.append(f(x0))
        xs.append(x0)
    return x0, fs, xs

def minimize(f, x0, step_size_fn, max_iter):
    df = num_grad(f)
    return gd(f, df, x0, step_size_fn, max_iter)

def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))

    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)

    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)

    x, fs, xs = gd(f, df, init, svm_min_step_size_fn, 50)
    return x, fs, xs

def acc(X,Y,th):
    d, n = X_test.shape
    cnt = 0
    for i in range(n):
        if (Y_test[0, i] * (np.dot(X_test[:, i].T, th[:-1, :]) + th[-1:, :])) > 0:
            cnt += 1

    return cnt / n * 100





data = load_fish_data("Fish.csv")
data = raw_data(data)
X,Y,X_test,Y_test = gen_data(data,0,1)
th,fs,xs = batch_svm_min(X,Y,0.1)

print(acc(X_test,Y_test,th))
