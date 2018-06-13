import sys
import numpy
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from random import shuffle
from csv import reader
import math
from sklearn.metrics import classification_report,roc_curve
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
data, labels = mnist["data"], mnist["target"]
labels[59999:70000]
data_full = np.column_stack((data,labels))
data_0_to_4 = data_full[np.where((data_full[:,-1] == 0) | (data_full[:,-1] == 1) | (data_full[:,-1] == 2) | (data_full[:,-1] == 3) | (data_full[:,-1] == 4))]

np.random.shuffle(data_0_to_4)
train_data, test_data = data_0_to_4[0:30000], data_0_to_4[30001:35735]

def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def softmax(x):
    #print(x)
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T

def train(x, y, w, b, lr=0.1, L2_reg=0.00):

        p_y_given_x = softmax(numpy.dot(x, w) + b)
        d_y = y - p_y_given_x
        #print(p_y_given_x)
        w += lr * numpy.dot(x.T, d_y) - lr * L2_reg * w
        b += lr * numpy.mean(d_y, axis=0)
        
        return w,b


def negative_log_likelihood(x, y, w, b):
        #softmax_activation = softmax(numpy.dot(x, w) + b)
        softmax_activation = softmax(numpy.dot(x, w) + b)
        cross_entropy = - numpy.mean(
            numpy.sum(y * numpy.log(softmax_activation) +
            (1 - y) * numpy.log(1 - softmax_activation),
                      axis=1))

        return cross_entropy


def predict(x, w, b):
        return softmax(numpy.dot(x, w) + b)


def encode(y):
    out_data = []
    for i in y:
        out_data.append(int(i))
    out = numpy.zeros((len(y), 5))
    out[numpy.arange(len(y)), out_data] = 1
    return out


squarer = lambda t: t/255
vfunc = np.vectorize(squarer)

def fit(learning_rate=0.01, n_epochs=10, n_in=784, n_out=5):

    x = vfunc(train_data[:,:-1])
    y = encode(train_data[:,[-1]])
    w = numpy.zeros((n_in, n_out),dtype=np.float128)
    b = numpy.zeros(n_out,dtype=np.float128)
    #w,b = train(x, y, w, b, lr=learning_rate, L2_reg=0.00)
    #cost = negative_log_likelihood(x, y, w, b)
    #print(cost,w,b)
    count = 0
    predicted = []
    original = []
    for epoch in range(n_epochs):
        w,b = train(x, y, w, b, lr=learning_rate, L2_reg=0.00)
        cost = negative_log_likelihood(x, y, w, b)
        print("epoch: ",count,"/",n_epochs)
        count += 1
        learning_rate *= 0.001

    correct = 0
    for i in range(len(test_data[:,:-1])):
        #print("predicting: ",i,"/",len(test_data[:,:-1]))
        temp = (predict(vfunc(test_data[:,:-1][i]), w, b))
        #print(temp)
        crt = np.argmax(temp, axis=0)
        original.append(int(test_data[:,-1][i]))
        predicted.append(crt)
        if(crt == test_data[:,-1][i]):
            correct += 1
    return(original,predicted,(correct/len(test_data[:,:-1])))


one,two,acc= fit()
print("Accuracy: ", acc*100)
print(classification_report(two,one))







