from utils import load_mnist
from sklearn.model_selection import train_test_split
from sklearn import svm
import requests
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def getParams():
    a,b,c = 1,1,1
    return a,b,c

def func(x):
    a,b,c = getParams()
    y = a + b*x + c*x**2
    return y


def grad(x):    
    _,b,c = getParams()
    gradient = b + 2*c*x
    return gradient


def updt(grad_func, x, a):
    x_new = x - a*grad_func(x)
    return x_new


def ex3a(x_train, y_train, kernel_type, cost):
    clf = SVC(kernel = kernel_type, C = cost)
    clf.fit(x_train, y_train)
    return clf

def ex3b(model, x_train, y_train, x_test, y_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_error = 1-accuracy_score(y_train, y_pred_train)
    test_error = 1-accuracy_score(y_test, y_pred_test)
    return y_pred_train,y_pred_test,train_error,test_error

def ex4(data):
    pass


if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.143)
