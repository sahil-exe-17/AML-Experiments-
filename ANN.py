!pip install -q ucimlrepo

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

import seaborn as sns

# 2. Load dataset from UCI ML Repository
cancer = fetch_ucirepo(id=17)

X = cancer.data.features
y = cancer.data.targets.iloc[:, 0]

# Remove missing values if present
data = pd.concat([X, y.rename("target")], axis=1).dropna()
X = data.drop("target", axis=1)
y = data["target"]

print("\nFirst 5 Records:")
print(X.head())

print("\nDataset Shape:", X.shape)

# 3. Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create ANN Model
ann = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

# 7. Train ANN
ann.fit(X_train, y_train)

# 8. Predict test data
y_pred = ann.predict(X_test)

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Algorithm: Artificial Neural Network")
print("Dataset: UCI Breast Cancer Wisconsin")
print("Total Records:", len(data))
print("Number of Features:", X.shape[1])
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))
