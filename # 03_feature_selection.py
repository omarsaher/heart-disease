# 03_feature_selection.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess
df = pd.read_csv("../data/heart_disease.csv").dropna()
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))
y = df["target"]

# RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X, y)
print("RFE selected features:", rfe.support_)

# Chi-Square
chi2_selector = SelectKBest(chi2, k=10)
chi2_selector.fit(X, y)
print("Chi2 scores:", chi2_selector.scores_)

# Random Forest Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
print("Random Forest Importances:", rf.feature_importances_)
