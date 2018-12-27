import numpy as np
import pandas as pd
from utils import load_mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Using python's functions
def cv(X,y,model,folds):    
    scores = cross_validate(model, X, y, cv=folds, scoring='accuracy', return_train_score=True)
    return np.mean(1 - scores['train_score']), np.mean(1 - scores['test_score'])

# Using a function we wrote 
# def cv(X,y,model,folds):
#     kf = KFold(n_splits = folds, random_state=1000)
#     mat_train = []
#     mat_val = []
#     for train_index, val_index in kf.split(X):
#         X_train_cv, X_val = X.iloc[train_index], X.iloc[val_index]
#         y_train_cv, y_val = y.iloc[train_index], y.iloc[val_index]
#         model.fit(X_train_cv, y_train_cv)
#         y_pred_train = model.predict(X_train_cv)
#         y_pred_val = model.predict(X_val)
#         train_error = 1-accuracy_score(y_train_cv, y_pred_train)
#         val_error = 1-accuracy_score(y_val, y_pred_val)
#         mat_train.append(train_error)
#         mat_val.append(val_error)
    
#     return np.mean(mat_train), np.mean(mat_val)

if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    
 

