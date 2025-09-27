# 06_hyperparameter_tuning.ipynb

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess
df = pd.read_csv("../data/heart_disease.csv").dropna()
df = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target", axis=1))
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Save best model
best_model = grid.best_estimator_
joblib.dump(best_model, "../models/final_model.pkl")
print("Model saved as ../models/final_model.pkl")
