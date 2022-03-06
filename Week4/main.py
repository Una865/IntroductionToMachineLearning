import numpy as np

def norm(th):
    return np.square(np.sum(th**2))

def sep(th,th0,data,labels):
   #returns margins
   # s1 - sum of all margins
   # s2 - minimum of all margins
   # s3 - maximum of all margins

   d, n = data.shape
   s1 = s2 = s3  = 0
   print(type(s1))
   normV = norm(th)

   for i in range(n):
       val = float(labels[0][i] * (np.dot(data[:, i].T, th) + th0) / normV)
       print(val*2**0.5)
       s1 += val
       if i == 0:
           s2 = val
           s3 = val
       else:
           s2 = min(s2, val)
           s3 = max(s3, val)


   return s1,s2,s3



data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
                 [1, 1, 2, 2,  2,  2,  2, 2]])
labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
blue_th = np.array([[0, 1]]).T
blue_th0 = -1.5
red_th = np.array([[1, 0]]).T
red_th0 = -2.5

##############################################################
#Implementation of gradient descent
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

