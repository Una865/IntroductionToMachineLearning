import numpy as np
import csv
import pandas as pd
from sklearn import svm

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
    dt =[]

    for i in range(n):

        keys = list(data[i].keys())
        data[i][keys[0]] = names[data[i][keys[0]]]
        vals = list(data[i].values())
        dt.append(vals)



    return dt



def gen_data(data,n1,n2):
    n= len(data)

    train = []
    X = []
    Y = []

    for i in range(n):
        if data[i][0] == n1 or data[i][0] == n2:
            X.append(data[i][1:])
            Y.append(data[i][0])

    n = len(X)
    idx = int(0.9*n)




    X_test = X[idx:]
    Y_test = Y[idx:]
    X = X[:idx]
    Y = Y[:idx]
    return X,Y,X_test,Y_test

    return X,Y,X_test,Y_test

data = load_fish_data('Fish.csv')
data = raw_data(data)
X,Y,X_test,Y_test = gen_data(data,0,1)
clf = svm.SVC()
clf.fit(X, Y)

n = len(X_test)
cnt = 0
for i in range(n):
    if clf.predict(X_test)[0]*Y_test[i] > 0:
        cnt+=1

print(cnt/n*100)
