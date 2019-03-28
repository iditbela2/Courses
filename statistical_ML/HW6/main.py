import pandas as pd
import numpy as np


def dist(x1,x2):
    distance = []
    for i in range(len(x1)):
        distance.append((x1[i]-x2[i])**2)
    return np.sqrt(sum(distance))

def k_means(data,k):
    # initialize centers randomly:
    columns = data.columns.values
    M = np.zeros((k,np.shape(data)[1])) # centers
    for i,col in enumerate(columns):
        for clust in range(k):
            M[clust,i] = random.uniform(data[col].min(),data[col].max())
    
    M_old = np.zeros((k,np.shape(data)[1])) # centers
    while(np.abs(np.max(M-M_old))>0): # convergence creterion
        M_old = np.copy(M)
        # calculate for each point the distance to the center of one of the k clusters:
        cluster_array = np.zeros((np.shape(data)[0],k))
        for i in range(np.shape(data)[0]):
            for k_ind in range(k):
                cluster_array[i,k_ind] = dist(data.loc[i,:].values,M_old[k_ind,:])
        assigned_clusters = np.argmin(cluster_array,axis = 1)
        # calculate new centers
        for i,col in enumerate(columns):
            for k_ind in range(k):
                M[k_ind,i] = np.sum(data.iloc[assigned_clusters == k_ind,i])/np.sum(assigned_clusters == k_ind)
#         print(M)
#         print(np.sum(np.min(cluster_array,axis = 1)))
  
    # calculate the objective function (M_old = M at that point)
    objective_function = np.sum(np.min(cluster_array,axis = 1))
    return objective_function # The total variance


if__"name"__==__"main"__:
	data = pd.read_csv('data.csv')
	