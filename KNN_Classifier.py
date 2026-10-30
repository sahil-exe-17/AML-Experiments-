# Experiment No. 7
# KNN Classification using Wine Dataset

import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ---------------------------------
# Step 1: Load Dataset
# ---------------------------------

wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)

y = wine.target


# ---------------------------------
# Step 2: Display First Five Records
# ---------------------------------

print("First Five Records")
print(X.head())


# ---------------------------------
# Step 3: Dataset Information
# ---------------------------------

print("\nDataset Shape:")
print(X.shape)

print("\nFeature Names:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(wine.target_names)


# ---------------------------------
# Step 4: Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ---------------------------------
# Step 5: Standardize the Data
# ---------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ---------------------------------
# Step 6: Create KNN Model
# ---------------------------------

model = KNeighborsClassifier(
    n_neighbors=5
)


# ---------------------------------
# Step 7: Train the Model
# ---------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------
# Step 8: Prediction
# ---------------------------------

y_pred = model.predict(X_test)


# ---------------------------------
# Step 9: Calculate Accuracy
# ---------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy =", accuracy)

print("Accuracy =", accuracy * 100, "%")


# ---------------------------------
# Step 10: Confusion Matrix
# ---------------------------------

print("\nConfusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ---------------------------------
# Step 11: Classification Report
# ---------------------------------

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=wine.target_names
    )
)
