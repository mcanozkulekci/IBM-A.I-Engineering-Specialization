# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 20:40:35 2020

@author: Code Channel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cust_df = pd.read_csv("data.csv",encoding='latin1')
print(cust_df.head(15))


#We need to eliminate non-numeric values so that Euclidean Distance function can process data.
df = cust_df.drop(columns=['InvoiceNo','StockCode','Description','InvoiceDate','Country'],axis=1)
print(df.head())

#Normalizing Data

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X=np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

#Modeling
#init: init method of the centroids
#k-means++ : Selects initial cluster centers for k-mean clustering in a more efficiently way

clusterNum = 4
c_means = KMeans(init='k-means++',n_clusters=clusterNum,n_init=12)
c_means.fit(X)
labels = c_means.labels_
print(labels)

df["Clus_km"] = labels
df.groupby('Clus_km').mean()



plt.scatter(X[:, :], X[:, :], c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Quantity', fontsize=18)
plt.ylabel('UnitPrice', fontsize=16)

plt.show()