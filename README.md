# AML Experiments

A collection of practical implementations for **Applied Machine Learning (AML)** using Python and scikit-learn. The repository covers data preprocessing, concept learning, classification, clustering, neural networks, and dimensionality reduction.

> These scripts are educational practicals. They demonstrate the core workflow of each technique and print model evaluation results where applicable.

## Repository contents

| File | Experiment / technique | Dataset used |
|---|---|---|
| `Data_Pre_Processing.py` | Missing-value handling, encoding, scaling, normalization, Sequential Forward Selection (SFS), Sequential Backward Selection (SBS) | Small manually created sample dataset |
| `Find_S.py` | Find-S concept-learning algorithm | Small manually created EnjoySport-style dataset |
| `Naive_Bayes.py` | Gaussian Naive Bayes classification | Wine dataset bundled with scikit-learn |
| `Decision_Tree.py` | Decision Tree classification (Gini criterion) | Wine dataset bundled with scikit-learn |
| `KNN_Classifier.py` | K-Nearest Neighbours classification | Wine dataset bundled with scikit-learn |
| `Random_Forest.py` | Random Forest classification | Wine dataset bundled with scikit-learn |
| `SVM.py` | Support Vector Machine classification with RBF kernel | Wine dataset bundled with scikit-learn |
| `ANN.py` | Artificial Neural Network using an MLP classifier | UCI Breast Cancer Wisconsin (Diagnostic), UCI ID 17 |
| `Kmeans_Clustering.py` | K-Means clustering and cluster-to-class evaluation | UCI Iris, UCI ID 53 |
| `PCA_AND_SVD.py` | Principal Component Analysis and Truncated SVD | UCI Wine Recognition, UCI ID 109 |

## Requirements

- Python 3.9 or later (a recent Python 3 version is recommended)
- pandas
- NumPy
- scikit-learn
- ucimlrepo (for the UCI-based ANN, K-Means, and PCA/SVD experiments)
- SciPy (for the K-Means cluster-label matching)
- Matplotlib (used by the dimensionality-reduction practical if visualisations are added)

Install the commonly used packages:

```bash
pip install pandas numpy scikit-learn ucimlrepo scipy matplotlib seaborn
```

Run a script from the repository directory:

```bash
python ANN.py
```

The UCI experiments download their datasets through the `ucimlrepo` package, so an internet connection is required the first time they are run. The scripts beginning with `!pip install` use notebook/Colab syntax; in a normal terminal-based Python script, install the package once using the command above and remove the `!` installation line if it causes a syntax error.

## Common machine-learning workflow

Most classification scripts follow these steps:

1. **Load data:** read the dataset and separate input features (`X`) from the target labels (`y`).
2. **Inspect data:** display sample rows, dimensions, feature names, or classes.
3. **Split data:** use `train_test_split` to create training and testing partitions. The training set is used to fit the model; the test set is held back for evaluation.
4. **Preprocess:** scale numerical features when the algorithm is sensitive to feature magnitudes.
5. **Create and train model:** instantiate an estimator and call `fit(X_train, y_train)`.
6. **Predict:** call `predict(X_test)`.
7. **Evaluate:** compare predictions with the true test labels using accuracy, a confusion matrix, and a classification report.

In this repository, `random_state=42` makes the train/test split or model initialisation repeatable. Where `stratify=y` is used, the split keeps approximately the same class proportions in both partitions.

---

## Experiment explanations

### 1. Data preprocessing and feature selection — `Data_Pre_Processing.py`

**Purpose:** Demonstrates common data preparation steps and two sequential feature-selection strategies.

**Dataset:** A five-row sample DataFrame created directly in the script. It contains Age, Salary, Gender, and Purchased columns, with a few missing numerical values.

**Code walkthrough:**

1. **Create a DataFrame:** the `data` dictionary is converted to a pandas DataFrame. `None` represents missing values.
2. **Impute missing values:** `SimpleImputer(strategy='mean')` replaces missing Age and Salary entries with the mean of their respective columns. The imputer is fitted and applied to those two numerical columns.
3. **Encode categories:** `LabelEncoder` converts Gender and Purchased text categories into integer codes so they can be used by the numerical model. The target column is encoded as well.
4. **Separate features and target:** `X = df.iloc[:, :-1]` selects all columns except the last one; `y = df.iloc[:, -1]` selects Purchased.
5. **Standardize:** `StandardScaler` transforms each feature to have approximately zero mean and unit variance.
6. **Normalize:** `MinMaxScaler` scales each feature to the range 0 to 1. The script prints both transformations to compare them.
7. **Split data:** 80% is used for training and 20% for testing.
8. **Create a KNN estimator:** `KNeighborsClassifier(n_neighbors=1)` predicts a sample based on its single nearest training neighbour.
9. **Sequential Forward Selection (SFS):** starts with no selected features. It tries each remaining feature, evaluates the KNN model, and adds the feature with the best accuracy. It repeats until two features have been selected.
10. **Sequential Backward Selection (SBS):** starts with all features. It tries removing each feature and permanently removes the one whose removal gives the best accuracy, repeating until two features remain.

**Important implementation note:** This is a small demonstration dataset, not a meaningful benchmark. The current feature-selection loops evaluate candidates on `X_test` and `y_test`; that means the test set influences feature selection. For a rigorous experiment, perform feature selection using training data only (preferably cross-validation), and use the test set once at the very end. Also, preprocessing is currently fitted before the train/test split; a production workflow should fit imputers and scalers on training data only, ideally through a scikit-learn Pipeline.

### 2. Find-S algorithm — `Find_S.py`

**Purpose:** Finds the most specific hypothesis that covers the positive training examples.

**Dataset:** A small manually entered six-attribute weather/EnjoySport-style dataset. The final column is the class label (`Yes` or `No`).

**Code walkthrough:**

1. **Training examples:** each row contains Sky, Temperature, Humidity, Wind, Water, Forecast, and the target class.
2. **Attribute names:** the `attributes` list contains the six input feature names.
3. **Initial hypothesis:** `['0'] * len(attributes)` creates the most-specific starting hypothesis. Here, `0` means no value has been learned yet.
4. **Process examples:** the loop checks the final value in each row. Negative examples (`No`) are ignored.
5. **Generalize on positive examples:** for each attribute, the first positive example supplies its value. If a later positive example has a different value, that hypothesis position becomes `?`, meaning any value is accepted.
6. **Display result:** the final list is the learned hypothesis.

**Example interpretation:** a hypothesis such as `['Sunny', 'Warm', '?', 'Strong', '?', '?']` means the learned concept requires Sunny weather and Warm temperature, while the remaining attributes can take any value.

**Limitation:** Find-S considers only positive examples. It does not use negative examples to eliminate hypotheses, and it assumes the examples are noise-free and the target concept can be represented by the given attribute constraints.

### 3. Gaussian Naive Bayes — `Naive_Bayes.py`

**Purpose:** Classifies wine samples into one of three wine classes using Bayes' theorem and a conditional-independence assumption.

**Dataset:** `load_wine()`, the Wine recognition dataset distributed with scikit-learn. It has 178 samples, 13 numerical chemical measurements, and three target classes. It is the classic UCI Wine dataset made available through scikit-learn's built-in dataset loader; this script does not download it with `ucimlrepo`.

**Code walkthrough:**

1. **Load data:** `load_wine()` returns feature values, target labels, feature names, and target class names.
2. **Build feature table:** a pandas DataFrame gives the 13 measurements readable column names. `wine.target` contains the numeric class labels.
3. **Inspect:** `head()`, `shape`, feature names, and target names show the dataset structure.
4. **Split:** 70% training and 30% testing, with `stratify=y` to preserve class proportions.
5. **Create model:** `GaussianNB()` models each feature's distribution within each class as a Gaussian (normal) distribution.
6. **Train:** `fit(X_train, y_train)` estimates class priors and feature distributions from the training data.
7. **Predict:** `predict(X_test)` selects the class with the highest posterior probability for each test sample.
8. **Evaluate:** accuracy gives the fraction of correct predictions; the confusion matrix shows correct and incorrect class assignments; the classification report gives precision, recall, and F1-score for each class.

**Key idea:** Naive Bayes applies Bayes' theorem while treating features as conditionally independent given the class. Gaussian Naive Bayes is suitable for continuous numerical features.

### 4. Decision Tree — `Decision_Tree.py`

**Purpose:** Builds a tree of feature-based decisions to classify wine samples.

**Dataset:** scikit-learn's built-in Wine dataset (178 samples, 13 features, 3 classes).

**Code walkthrough:**

1. **Load and organise:** `load_wine()` provides the feature matrix and class labels. A DataFrame adds feature names.
2. **Inspect:** print the first five rows, dimensions, feature names, and class names.
3. **Split:** 70% training and 30% testing; `stratify=y` preserves class proportions.
4. **Create tree:** `DecisionTreeClassifier(criterion='gini', random_state=42)` uses Gini impurity to choose splits. A split is useful when it makes the resulting groups more class-pure.
5. **Train:** `fit()` builds the decision tree from the training examples.
6. **Predict:** `predict()` assigns a wine class to each test record.
7. **Evaluate:** accuracy, confusion matrix, and classification report summarise performance.

**Key idea:** A decision tree repeatedly asks questions about feature values. Each internal node is a decision, each branch is an outcome, and each leaf gives a predicted class.

### 5. K-Nearest Neighbours — `KNN_Classifier.py`

**Purpose:** Classifies a wine sample using the labels of its nearest training samples.

**Dataset:** scikit-learn's built-in Wine dataset (178 samples, 13 features, 3 classes).

**Code walkthrough:**

1. **Load and inspect:** load the Wine data, attach feature names, and display its dimensions and target classes.
2. **Split:** 70% training and 30% testing with stratification.
3. **Scale features:** `StandardScaler` is fitted on training data and then applied to both training and test data. This matters because KNN relies on distances; large-scale features could otherwise dominate.
4. **Create model:** `KNeighborsClassifier(n_neighbors=5)` uses the five closest training samples.
5. **Train:** `fit()` stores the scaled training examples and labels for neighbour lookup.
6. **Predict:** for each test sample, KNN finds its five nearest training samples and predicts the majority class.
7. **Evaluate:** accuracy, confusion matrix, and classification report show how well the predictions match the true labels.

**Key idea:** KNN is a distance-based, instance-based algorithm. The value of K affects the bias/variance trade-off; this practical fixes K at 5.

### 6. K-Means clustering — `Kmeans_Clustering.py`

**Purpose:** Groups Iris flower measurements into three clusters and compares those clusters with known flower classes for evaluation.

**Dataset:** UCI Iris dataset fetched using `fetch_ucirepo(id=53)`. It contains 150 records, four numerical measurements (sepal length, sepal width, petal length, petal width), and three species.

**Code walkthrough:**

1. **Install/import:** `ucimlrepo` fetches the UCI dataset. pandas and NumPy handle data; scikit-learn provides splitting, scaling, and K-Means; SciPy's assignment solver is used to match clusters to class labels.
2. **Load data:** `iris.data.features` becomes `X`; the first target column becomes `y`.
3. **Encode labels:** `LabelEncoder` converts the three species names into integer labels. These labels are not used to fit K-Means; they are kept for evaluation.
4. **Split:** 80% training and 20% testing, stratified by the known species labels.
5. **Scale:** `StandardScaler` is fitted on training data only, then applied to the test data. Scaling is important because K-Means uses distances.
6. **Fit K-Means:** `KMeans(n_clusters=3, random_state=42, n_init=10)` learns three centroids from the scaled training data. Each point is assigned to its nearest centroid.
7. **Predict clusters:** the fitted model assigns both training and test samples to cluster IDs 0, 1, or 2.
8. **Match IDs to classes:** cluster numbers are arbitrary and do not inherently mean Setosa, Versicolor, or Virginica. The script creates a training confusion matrix and uses `linear_sum_assignment` to find the one-to-one mapping that maximises matches.
9. **Evaluate:** the mapped test cluster IDs are compared with the true test labels using accuracy.

**Important interpretation:** K-Means is unsupervised. The class labels are used only after clustering to align arbitrary cluster IDs for a meaningful comparison. The reported accuracy is a cluster-to-class agreement measure, not ordinary supervised classification accuracy. A one-to-one mapping also assumes the three clusters can be paired with the three species.

### 7. Artificial Neural Network — `ANN.py`

**Purpose:** Trains a feed-forward neural network to classify breast cancer diagnostic records as benign or malignant.

**Dataset:** UCI Breast Cancer Wisconsin (Diagnostic), fetched using `fetch_ucirepo(id=17)`. The dataset has 569 records and 30 numerical input features; the target has two diagnosis classes.

**Code walkthrough:**

1. **Install/import:** `ucimlrepo` downloads the dataset. pandas/NumPy organise the data, and scikit-learn provides splitting, scaling, the neural network, and evaluation metrics.
2. **Load data:** `cancer.data.features` supplies the input features; the first target column supplies the diagnosis.
3. **Remove missing rows:** features and target are combined into one DataFrame, and `dropna()` removes rows containing missing values. The features and target are then separated again.
4. **Encode labels:** `LabelEncoder` converts the two text diagnosis classes into numeric labels.
5. **Split:** 80% training and 20% testing, using stratification to preserve class proportions.
6. **Scale:** `StandardScaler` is fitted on the training features and applied to the test features. Neural networks generally train more reliably when input features have comparable scales.
7. **Build ANN:** `MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', ...)` creates two hidden layers with 16 and 8 neurons. ReLU is the hidden-layer activation function; Adam optimises the weights. `max_iter=1000` sets the iteration limit, and `early_stopping=True` reserves part of the training data for validation and can stop training when validation performance stops improving.
8. **Train:** `fit()` learns weights and biases through backpropagation.
9. **Predict:** `predict(X_test)` returns the predicted diagnosis for each test record.
10. **Evaluate:** accuracy and the classification report summarise test performance using precision, recall, and F1-score.

**Note about the current file:** confusion-matrix functions are imported, but the script currently does not print or plot a confusion matrix. The report is the evaluation output actually produced by this file.

**Key idea:** An ANN is made of connected neurons arranged in layers. During training, the network adjusts connection weights to reduce prediction error.

### 8. PCA and SVD — `PCA_AND_SVD.py`

**Purpose:** Reduces the Wine dataset from 13 numerical features to two dimensions using PCA and Truncated SVD, then reports how much variance the two components explain.

**Dataset:** UCI Wine Recognition dataset fetched using `fetch_ucirepo(id=109)`. It contains 178 records and 13 numerical chemical features. The target column is loaded for reference but is not used in the transformations.

**Code walkthrough:**

1. **Install/import:** `ucimlrepo` retrieves the dataset; NumPy/pandas handle the arrays and tables; scikit-learn provides scaling, PCA, and Truncated SVD.
2. **Load:** `wine.data.features` is the feature matrix `X`; the first target column is `y`.
3. **Inspect:** print the first five feature rows and original matrix dimensions.
4. **Scale:** `StandardScaler` standardises all 13 features. This prevents measurements with larger numerical ranges from dominating the analysis.
5. **Apply PCA:** `PCA(n_components=2)` finds two orthogonal directions that retain as much variance as possible in the centred data. `fit_transform()` learns the components and projects each wine record into two coordinates.
6. **Read PCA variance:** `explained_variance_ratio_` gives the fraction of total variance represented by each principal component. Their sum, multiplied by 100, is the percentage retained by the two components.
7. **Apply SVD:** `TruncatedSVD(n_components=2)` computes a two-component truncated singular-value decomposition and transforms the scaled matrix into two coordinates.
8. **Read SVD variance:** its `explained_variance_ratio_` values are printed and summed for a two-component variance summary.

**Key ideas:**
- **PCA (Principal Component Analysis)** is a dimensionality-reduction method that finds directions of maximum variance in the data.
- **SVD (Singular Value Decomposition)** factorises a matrix into singular vectors and singular values. Truncated SVD keeps only a selected number of components to form a lower-dimensional representation.

**Important implementation note:** this script currently prints the transformed shapes and explained-variance values; it does not create plots or compare predictive accuracy. PCA and SVD are dimensionality-reduction methods, not classifiers. The target labels are not used to train either transformation.

### 9. Support Vector Machine — `SVM.py`

**Purpose:** Classifies wine samples by finding decision boundaries that separate the three classes with a large margin.

**Dataset:** scikit-learn's built-in Wine dataset (178 samples, 13 numerical features, 3 classes).

**Code walkthrough:**

1. **Load and inspect:** load the Wine dataset and create a DataFrame with readable feature names.
2. **Split:** 70% training and 30% testing, preserving class proportions with stratification.
3. **Scale:** `StandardScaler` is fitted on training data and applied to the test data. Scaling is important for SVM because feature magnitudes affect distances and the RBF kernel.
4. **Create model:** `SVC(kernel='rbf', random_state=42)` creates a Support Vector Classifier with a Radial Basis Function kernel. The RBF kernel can model non-linear decision boundaries.
5. **Train:** `fit()` learns the separating boundaries from the training examples.
6. **Predict:** `predict()` assigns a class to each test sample.
7. **Evaluate:** accuracy, confusion matrix, and classification report summarise test performance.

**Key idea:** Support vectors are the training points closest to the decision boundary. They play a central role in defining the classifier.

### 10. Random Forest — `Random_Forest.py`

**Purpose:** Classifies wine samples using an ensemble of decision trees.

**Dataset:** scikit-learn's built-in Wine dataset (178 samples, 13 features, 3 classes).

**Code walkthrough:**

1. **Load and inspect:** obtain the Wine feature matrix and target labels and display dataset information.
2. **Split:** 70% training and 30% testing, with stratification.
3. **Create forest:** `RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)` creates an ensemble of 100 decision trees. Gini impurity is used for tree split quality.
4. **Train:** `fit()` builds the trees using bootstrapped samples and feature subsets.
5. **Predict:** each tree votes for a class; the forest returns the class with the most votes.
6. **Evaluate:** accuracy, confusion matrix, and classification report show test-set performance.

**Key idea:** Random Forest combines many decision trees. Combining their predictions often produces a more stable model than relying on a single tree.

---

## Important concepts used across the repository

| Term | Simple meaning |
|---|---|
| Feature (`X`) | An input measurement used to make a prediction |
| Target (`y`) | The correct class or value the model is trying to predict |
| Training set | Data used to learn model parameters |
| Test set | Separate data used to estimate performance on unseen examples |
| Imputation | Filling in missing values |
| Label encoding | Converting category labels into numbers |
| Standardization | Rescaling values to approximately zero mean and unit variance |
| Normalization | Rescaling values, commonly to the range 0–1 |
| Accuracy | Fraction of predictions that are correct |
| Confusion matrix | Table comparing actual and predicted classes |
| Precision | Of the samples predicted as a class, the fraction that truly belong to it |
| Recall | Of the samples truly in a class, the fraction correctly found |
| F1-score | Harmonic mean of precision and recall |
| Dimensionality reduction | Representing data with fewer features/components |

## Dataset source notes

- **UCI Iris (ID 53):** used by `Kmeans_Clustering.py`.
- **UCI Breast Cancer Wisconsin (Diagnostic) (ID 17):** used by `ANN.py`.
- **UCI Wine Recognition (ID 109):** used by `PCA_AND_SVD.py`.
- **Wine dataset via scikit-learn:** used by `Naive_Bayes.py`, `Decision_Tree.py`, `KNN_Classifier.py`, `Random_Forest.py`, and `SVM.py`. It is available locally through `sklearn.datasets.load_wine` after scikit-learn is installed.
- **Manually created sample data:** used by `Data_Pre_Processing.py` and `Find_S.py`.

UCI repository: https://archive.ics.uci.edu/

## Suggested viva revision

- **Preprocessing:** Why do we handle missing values? What is the difference between standardization and normalization? Why encode categorical data?
- **Find-S:** What is a hypothesis? What does `?` mean? Why are negative examples ignored?
- **Naive Bayes:** State Bayes' theorem. What is the naive independence assumption? Why is it called Gaussian Naive Bayes?
- **Decision Tree:** What is Gini impurity? What are root, decision, and leaf nodes?
- **KNN:** How does K affect prediction? Why is scaling important? What distance is commonly used?
- **K-Means:** What is a centroid? How is K selected? Why are cluster IDs not class labels?
- **ANN:** What are input, hidden, and output layers? What is an activation function? What is backpropagation?
- **PCA/SVD:** What is dimensionality reduction? What is explained variance? How are PCA and SVD related?
- **SVM:** What is a hyperplane? What are support vectors? What does the RBF kernel do?
- **Random Forest:** What is an ensemble? How does voting work? How is a Random Forest different from one Decision Tree?

## Notes and limitations

- These are practical teaching scripts, not production-ready pipelines.
- Results can vary with library versions and environment details.
- For fair model evaluation, keep the test set separate from preprocessing choices and feature selection. Fit preprocessing steps only on training data.
- The UCI-fetching scripts require internet access.
- The repository contains standalone scripts; each file can be run independently after installing its dependencies.

## License

No license is specified in this repository at the time this README was prepared. Add a license file if you want to define how others may use, modify, and distribute this work.
