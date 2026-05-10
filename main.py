# Loan Default Prediction (Credit Risk Modeling)

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("credit_risk_dataset.csv")

print("Dataset Loaded Successfully\n")

print(df.head())

# =========================
# DATASET INFORMATION
# =========================

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# =========================
# HANDLE MISSING VALUES
# =========================

df.fillna(df.mean(numeric_only=True), inplace=True)

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

le = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset:\n")
print(df.head())

# =========================
# EXPLORATORY DATA ANALYSIS
# =========================

# Loan Default Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=df)
plt.title("Loan Default Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# DEFINE FEATURES & TARGET
# =========================

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# =========================
# MAKE PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:\n")
print(accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAMPLE PREDICTION
# =========================

sample_prediction = model.predict([X.iloc[0]])

print("\nSample Prediction:")
print(sample_prediction)

if sample_prediction[0] == 1:
    print("Customer is likely to default on loan")
else:
    print("Customer is NOT likely to default on loan")