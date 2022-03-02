import tensorflow as tf
import numpy as np
import lab3_data
import csv
import pandas as pd
import openpyxl
from sklearn.feature_extraction.text import HashingVectorizer
import nltk
import re
import heapq
import numpy as np


    #data - nXd

def proces_column(column):
    MEAN = np.mean(column)
    STD = np.std(column)
    column = (column-MEAN)/STD
    return column

def process_data(data):
    displacement = data[:,1]
    horsepower = data[:,2]
    weight = data[:,3]
    acceleration = data[:,[4]]
    model_year = data[:,[5]]
    origin = data[:,[6]]

    displacement = proces_column(displacement)
    horsepower = proces_column(horsepower)
    weight = proces_column(weight)
    #print(weight)
    data[:,1] = displacement
    data[:, 2] = horsepower
    data[:, 3] = weight
    return data

def one_hot_origin(data):

    USA = []
    europe = []
    asia = []
    row = 0
    col = 0
    n,d = data.shape
    for i in range(n):
        if data[i][6] == 1:
            USA.append([1])
        else:
            USA.append([0])
        if data[i][6]==2:
            europe.append([1])
        else:
            europe.append([0])
        if data[i][6] == 3:
            asia.append([1])
        else:
            asia.append([0])

    data  = np.append(data,USA,axis = 1)
    data = np.append(data, europe, axis=1)
    data = np.append(data, asia, axis=1)

    data = np.delete(data,6,1)
    return data

def process_text(data, column):

    text = ""
    n, d = data.shape
    for i in range(n):
        text+=column[i]
        text+= " "
    dataText = nltk.sent_tokenize(text)

    for i in range(len(dataText)):
        dataText[i] = dataText[i].lower()
        dataText[i] = re.sub(r'\W', ' ', dataText[i])
        dataText[i] = re.sub(r'\s+', ' ', dataText[i])
    word2count = {}
    for dataT in dataText:
        words = nltk.word_tokenize(dataT)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word]+=1

    #freq_words = heapq.nlargest(100,word2count,key = word2count.get)
    n2 = len(word2count)
    #new_col = np.zeros((n,n2))
    X =[]
    for i in range(n):
        vector = []
        for word in word2count.keys():
            if word in nltk.word_tokenize(data[i][6]):
                vector.append(1)
            else:
                vector.append(0)
        X.append(vector)
    new_col =  np.asarray(X)
    data = np.delete(data,6,1)
    data = np.append(data,new_col,axis = 1)
    return data


def prepare_data(filename = 'auto-mpg.tsv'):

    df = pd.read_csv('auto-mpg.tsv', sep="\t")
    data = df.to_numpy()
    np.random.shuffle(data)
    labels = data[:, [0]]
    data = np.delete(data,0,1)
    return data, labels

def averaged_perceptron(data,labels, params = {}, hack = None):
    T = params.get('T',10000)

    n,d = data.shape
    th = np.zeros((d,1))
    th0 = np.zeros((1,1))

    ths = np.zeros((d, 1))
    th0s = np.zeros((1, 1))

    for t in range(T):
        for i in range(n):
            X = data[i,:]
            Y = labels[i,0]
            if Y*(np.dot(X, th)+th0) <= 0:
                th[:, 0] = th[:,0] + X.T*Y
                th0 = th0 + Y
            ths +=th
            th0s+=th0
    return (ths/(n*T),th0s/(n*T))

def score(data_test,labels_test,th,th0):
    scr = 0

    n,d = data_test.shape
    for i in range(n):
        X = data[i,:]
        Y = labels_test[i,0]
        #print(np.dot(X, th)+th0)
        if Y*(np.dot(X, th)+th0) >= 0:
            scr+=1

    print(scr*100.0/n)



data, labels = prepare_data()
data =process_data(data)
data = one_hot_origin(data)
data = process_text(data,data[:,6])

data_test = data[353:]
data = data[:352]

labels_test = labels[353:]
labels = labels[:352]

df = pd.DataFrame(data.copy())
filepath = 'my_excel_file.xlsx'
df.to_excel(filepath, index=False)

th,th0 = averaged_perceptron(data,labels)
score(data_test,labels_test,th,th0)






