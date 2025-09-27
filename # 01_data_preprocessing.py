# 01_data_preprocessing.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/heart_disease.csv")
print("Original shape:", df.shape)

# Handle missing values
df = df.dropna()
print("After removing missing values:", df.shape)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))
y = df["target"]

print("Final features shape:", X.shape)
