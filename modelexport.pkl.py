import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("../data/heart_disease.csv").dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline (scaler + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Save pipeline as .pkl
joblib.dump(pipeline, "../models/final_pipeline.pkl")
print("✅ Pipeline saved as ../models/final_pipeline.pkl")
