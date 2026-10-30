!pip install -q ucimlrepo

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

# 2. Load dataset from UCI ML Repository
wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets.iloc[:, 0]

print("UCI Wine Dataset Loaded Successfully!")
print("\nFirst 5 Records:")
print(X.head())

print("\nOriginal Dataset Shape:", X.shape)

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n========== PCA RESULTS ==========")
print("Original Features:", X.shape[1])
print("Reduced Features:", X_pca.shape[1])
print("PCA Transformed Shape:", X_pca.shape)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total Explained Variance:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# 5. Apply SVD
svd = TruncatedSVD(n_components=2, random_state=42)

X_svd = svd.fit_transform(X_scaled)

print("\n========== SVD RESULTS ==========")
print("Original Features:", X.shape[1])
print("Reduced Features:", X_svd.shape[1])
print("SVD Transformed Shape:", X_svd.shape)

print("\nSVD Explained Variance Ratio:")
print(svd.explained_variance_ratio_)

print("Total SVD Explained Variance:",
      round(sum(svd.explained_variance_ratio_) * 100, 2), "%")

