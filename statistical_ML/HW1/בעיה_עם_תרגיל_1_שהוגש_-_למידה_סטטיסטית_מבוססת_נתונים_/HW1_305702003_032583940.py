
# coding: utf-8

# # Q2 - Linear Regression

# In[347]:


import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# ## a

# In[348]:


df = pd.read_csv('parkinsons_updrs_data.csv')


# In[349]:


df.head()
df.columns


# ## b

# **NOTES TO MYSELF**
# 
# The data contains data for 42 patients (subjects). For these <br/>
# 42 patients we have 5,875 voice recordings. 
# 
# **general data:**<br/>
# subject.ID - Integer that uniquely identifies each subject<br/>
# age - Subject age<br/>
# sex - Subject gender '0' - male, '1' - female<br/>
# test_time - Time since recruitment into the trial. The integer part is the <br/>
# number of days since recruitment.<br/>
# 
# **what we want to predict:**<br/>
# motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated<br/>
# total_UPDRS - Clinician's total UPDRS score, linearly interpolated<br/>
# 
# **measuremets (16 biomedical voice measures):**<br/>
# Jitter.Per,Jitter.Abs,Jitter.RAP,Jitter.PPQ5,Jitter.DDP - Several measures of <br/>
# variation in fundamental frequency<br/>
# Shimmer,Shimmer.dB,Shimmer.APQ3,Shimmer.APQ5,Shimmer.APQ11,Shimmer.DDA - <br/>
# Several measures of variation in amplitude<br/>
# NHR,HNR - Two measures of ratio of noise to tonal components in the voice<br/>
# RPDE - A nonlinear dynamical complexity measure<br/>
# DFA - Signal fractal scaling exponent<br/>
# PPE - A nonlinear measure of fundamental frequency variation <br/>

# In[350]:


df.describe()


# **age of subjects** - the mean age of the subjects is 64.8 years old, with a standard deviation of 8.8 years. The youngest subject is 36 years old while the oldest is 85.  
# 
# **gender of subjects** - since 0 represents a male and 1 - a female, we can see by the "mean sex", that about 32% of the subjects are females, meaning the majuraty (68%) are males. 

# ## c

# In[351]:


df_6 = df[['age','sex','Jitter.Per','Shimmer','NHR','PPE','motor_UPDRS']]


# In[352]:


scatter_plot = pd.plotting.scatter_matrix(df_6, alpha=0.8,s=30,figsize=(15,15),
                                          ax=None, grid=True, diagonal='hist',
                                          marker='o', density_kwds=None, 
                                          hist_kwds={'bins':20,'edgecolor':'black'},
                                          range_padding=0.05)


# ## d

# In[353]:


def leastSquares(X,y): #X and y are numpy arrays 
    
    estimators = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),
                                     np.transpose(X)),y)
    
    return estimators


# ## e

# In[354]:


import numpy as np
import statsmodels.api as sm

X = df_6[['age','sex','Jitter.Per','Shimmer','NHR','PPE']].values
y = df_6['motor_UPDRS'].values
X2 = sm.add_constant(X)
# np.shape(X)
# np.shape(y)

w = leastSquares(X2,y)


# ## f

# In[355]:


# python's linear regression (sklearn)
# from sklearn import linear_model
# import statsmodels.api as sm
# X2 = sm.add_constant(X)
# reg = linear_model.LinearRegression()
# reg2 = reg.fit(X2, y)                                      
# print(reg2.coef_)

# python's linear regression (statsmodels)
import statsmodels.api as sm

X2 = sm.add_constant(X)
reg = sm.OLS(y, X2)
reg2 = reg.fit()


# Yes, I got the same values. <br/>
# The following table summarizes the results.

# In[356]:


data = {'My_model': w,
        'Python_model': reg2.params}
table = pd.DataFrame(data)

# The following table compares both models' estimators
print(table)


# ## g

# We can use a t-test, provided by the the linear regression model from statsmodels:

# In[357]:


# sig_level = 0.01
sig_level = 0.01
for i in range(len(reg2.pvalues)-1):
    if (reg2.pvalues[i+1]<sig_level):
        print("Rejecting the null hypothesis with 0.01 significance level of 0.01 for parameter: " 
              + df_6.columns[i])

# sig_level = 0.001
sig_level = 0.001
for i in range(len(reg2.pvalues)-1):
    if (reg2.pvalues[i+1]<sig_level):
        print("Rejecting the null hypothesis with 0.01 significance level of 0.001 for parameter: " 
              + df_6.columns[i])
               
# reg2.summary()
# reg2.params


# # Q3 - Logistic Regression

# ## a

# In[358]:


from IPython.display import Math
Math(r'l(w) = argmax_w \sum_{i=1}^{m} log(P(y_i|x_i)) = ' +
     'argmax_w \sum_{i=1}^{m} log(\frac{1}{1+exp(-y_i w^T x_i) })')


# ## b

# In[359]:


Math(r'argmax_w \sum_{i=1}^{m} log(\frac{1}{1+exp(-y_i w^T x_i) }) = ' +
     'argmax_w \sum_{i=1}^{m} log(1) - log(1+exp(-y_i w^T x_i)) = ' +
     'argmin_w \sum_{i=1}^{m} log(1+exp(-y_i w^T x_i))')  


# # Q4 - Multiple Logistic Regression

# ## a

# In[360]:


from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target


# ## b

# In[361]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)


# ## c

# In[362]:


# change data
from copy import copy, deepcopy

# 1
y1 = deepcopy(y_train)

# 1
y1[y1 == 1] = -1# turn vers(1) ad virg(2) to -1
y1[y1 == 2] = -1
y1[y1 == 0] = 1 #turn setosa (0) to 1

# train a logistic regression classfier
from sklearn.linear_model import LogisticRegression
clf_1 = LogisticRegression(fit_intercept = True).fit(X_train, y1)
#instead of adding a constat to X use fit_intercept


# ## d

# In[363]:


# 2
y2 = deepcopy(y_train)

y2[y2 == 0] = -1
y2[y2 == 2] = -1
# y2[y2 == 1] = 1 

# train a logistic regression classfier
from sklearn.linear_model import LogisticRegression
clf_2 = LogisticRegression(fit_intercept = True).fit(X_train, y2)


# In[364]:


# 3 
y3 = deepcopy(y_train)

y3[y3 == 0] = -1
y3[y3 == 1] = -1
y3[y3 == 2] = 1

# train a logistic regression classfier
from sklearn.linear_model import LogisticRegression
clf_3 = LogisticRegression(fit_intercept = True).fit(X_train, y3)


# ## e

# In[365]:



def one_versus_rest(clf_1,clf_2,clf_3,X):
    # the order of the classes classified we can get from clf_1.classes_
    pr_1 = clf_1.predict_proba(X) 
    pr_2 = clf_2.predict_proba(X)
    pr_3 = clf_3.predict_proba(X)
    max_prob = np.array([pr_1[:,1],pr_2[:,1],pr_3[:,1]])
    y_predicted = np.argmax(max_prob,axis=0)
    return y_predicted
    


# ## f

# In[366]:


from sklearn.metrics import confusion_matrix

y_pred = one_versus_rest(clf_1,clf_2,clf_3,X_test)
y_true = deepcopy(y_test)

print(confusion_matrix(y_true, y_pred))


# ## g

# In[367]:


# create a data frame from the test data 
data = {iris.feature_names[0]: X_test[:,0],
        iris.feature_names[1]: X_test[:,1],
        iris.feature_names[2]:X_test[:,2],
        iris.feature_names[3]:X_test[:,3],
        'y_true':y_true,
        'y_pred':y_pred}
df = pd.DataFrame(data)


df.head(10)


# we can see that all mistakes are betweenn class 1 and class 2 <br/>
# (plot all to see all). if we look at the means of the properties we <br/>
# can see that class 1 and 2 are indeed the most similar.

# In[368]:


df.groupby('y_true').mean()


# Lets plot for example the scatter of sepal width VS. sepal lenngth:

# In[369]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,5))
fig.subplots_adjust(hspace=0.1, wspace=0.3)

ax = fig.add_subplot(1, 2, 1)
plt.scatter(df[iris.feature_names[2]], df[iris.feature_names[3]], c=df['y_true'])
plt.title('\ny_true\n',fontsize=20)
plt.ylabel(iris.feature_names[3],labelpad=20)
plt.xlabel(iris.feature_names[2],labelpad=20)
circle = plt.Circle((4.5, 1.5), 0.3, color='r', fill=False)
ax.add_artist(circle)

ax = fig.add_subplot(1, 2, 2)
plt.scatter(df[iris.feature_names[2]], df[iris.feature_names[3]], c=df['y_pred'])
plt.title('\ny_predicted\n',fontsize=20)
plt.ylabel(iris.feature_names[3],labelpad=20)
plt.xlabel(iris.feature_names[2],labelpad=20)

circle = plt.Circle((4.5, 1.5), 0.3, color='r', fill=False)
ax.add_artist(circle)

plt.show();


# we can see that group 0 (purple) is far from the other two groups - 1 and 2. <br/>
# and group 1 and 2 have a "common area" of similar length and width

#  
