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