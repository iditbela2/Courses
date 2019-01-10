import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

#adjust_labels_to_binary function get as arguments the training set target y_train as nparry,and a target class value as string
#it reteruns an nparry with the same shape as y_train with only binary labels: 1 for the target class value and -1 otherwise
def adjust_labels_to_binary(y_train, target_class_value):
    y_train_adj = np.copy(y_train)
    y_train_adj[y_train != target_class_value] = -1
    y_train_adj[y_train == target_class_value] = 1
    return y_train_adj

#one_vs_rest function gets as arguments x_train and y_train both as nparrays, and target_class_value as string
#it first binarize y_train according to target_class_value via the function adjust_labels_to_binary
#it returns a logistic regression model object trained on x_train and y_binarized     
def one_vs_rest(x_train, y_train, target_class_value):
    y_train_binarized = adjust_labels_to_binary(y_train, target_class_value)
    clf = LogisticRegression(fit_intercept=True).fit(x_train,y_train_binarized)
    return clf

#binarized_confusiom_matrix gets as arguments X, y_binarized as nparrays, the appropraite one_vs_rest_model as a model object
#and prob_threshold value
#it utilizes one_vs_rest model and predicted probabilities and the prob_threshold to predict
#y_pred on X
#it comparse it to the  y_binarized and 
#return an nparray of the appropriate confusion matrix as follows:
#[TP, FP
#FN, TN]    
def binarized_confusion_matrix(X, y_binarized, one_vs_rest_model, prob_threshold):
    from sklearn.metrics import confusion_matrix
    y_predicted_probabilities = one_vs_rest_model.predict_proba(X)[:,1] #class 1
    ind = y_predicted_probabilities>=prob_threshold
    y_predicted = -1*np.ones(np.shape(y_binarized))
    y_predicted[ind] = 1
    TN, FP, FN, TP = confusion_matrix(y_binarized,y_predicted, labels=[-1,1]).ravel() 
    # reorder the confusion matrix to display as asked 
    return TP,FN,FP,TN


#micro_avg_precision gets as arguments X, y as nparrays, 
#all_target_class_dict a dictionary with key class value as string with value per key of the approprite one_vs_rest model
#prob_threshold the probability that if greater or equal to the prediction is 1, otherwise -1
#returns the micro average precision
def micro_avg_precision(X, y, all_target_class_dict, prob_threshold):
    target = [0,1,2]
    total_TP = []
    total_FP = []
    for t in target:        
        y_binarized = adjust_labels_to_binary(y, target_class_value=t)
        one_vs_rest_model = all_target_class_dict[np.str(t)]   
        TP,FN,FP,TN = binarized_confusion_matrix(X,y_binarized, one_vs_rest_model, prob_threshold)
        total_TP.append(TP)
        total_FP.append(FP)
    MAP = np.sum(total_TP)/(np.sum(total_TP) + np.sum(total_FP))
    return MAP

def micro_avg_recall(X, y, all_target_class_dict, prob_threshold):
    target = [0,1,2]
    total_TP = []
    total_FN = []
    for t in target:        
        y_binarized = adjust_labels_to_binary(y, target_class_value=t)
        one_vs_rest_model = all_target_class_dict[np.str(t)]   
        TP,FN,FP,TN = binarized_confusion_matrix(X,y_binarized, one_vs_rest_model, prob_threshold)
        total_TP.append(TP)
        total_FN.append(FN)
    MAR = np.sum(total_TP)/(np.sum(total_TP) + np.sum(total_FN))
    return MAR

def micro_avg_false_positve_rate(X, y, all_target_class_dict, prob_threshold):
    target = [0,1,2]
    total_TN = []
    total_FP = []
    for t in target:        
        y_binarized = adjust_labels_to_binary(y, target_class_value=t)
        one_vs_rest_model = all_target_class_dict[np.str(t)]   
        TP,FN,FP,TN = binarized_confusion_matrix(X,y_binarized, one_vs_rest_model, prob_threshold)
        total_TN.append(TN)
        total_FP.append(FP)
    MAFPR = np.sum(total_FP)/(np.sum(total_TN) + np.sum(total_FP))
    return MAFPR

def f_beta(precision, recall, beta):
    fBeta = (1+beta**2)*(precision*recall)/(precision*beta**2 + recall)
    return fBeta

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98, test_size=0.3)
