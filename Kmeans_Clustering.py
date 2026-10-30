!pip install -q ucimlrepo

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# 2. Load dataset directly from UCI ML Repository
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets.iloc[:, 0]

print("\nFirst 5 Records:")
print(X.head())

# 3. Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 4. Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

# 7. Predict test data clusters
train_clusters = kmeans.predict(X_train_scaled)
test_clusters = kmeans.predict(X_test_scaled)

# 8. Match clusters with actual classes using training data
cm = confusion_matrix(
    y_train,
    train_clusters,
    labels=[0, 1, 2]
)

row_ind, col_ind = linear_sum_assignment(-cm)

cluster_to_class = {
    cluster: actual_class
    for actual_class, cluster in zip(row_ind, col_ind)
}

y_pred = np.array([cluster_to_class[c] for c in test_clusters])

# 9. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Algorithm: K-Means Clustering")
print("Dataset: UCI Iris Dataset")
print("Total Records:", len(X))
print("Number of Features:", X.shape[1])
print("Number of Clusters: 3")
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

