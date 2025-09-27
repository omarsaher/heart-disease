# 05_unsupervised_learning.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load and preprocess
df = pd.read_csv("../data/heart_disease.csv").dropna()
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
print("KMeans cluster labels:", np.unique(clusters))

# Hierarchical Clustering Dendrogram
plt.figure()
sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
