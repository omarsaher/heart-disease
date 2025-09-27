# 02_pca_analysis.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load and preprocess
df = pd.read_csv("../data/heart_disease.csv").dropna()
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))

# PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot cumulative explained variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.show()
