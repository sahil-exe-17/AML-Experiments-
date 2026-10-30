import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# =====================================================
# STEP 1: CREATE SAMPLE DATASET
# =====================================================

data = {
    'Age': [25, 30, None, 35, 40],
    'Salary': [50000, 60000, 65000, None, 70000],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Original Dataset\n")
print(df)


# =====================================================
# STEP 2: HANDLE MISSING VALUES
# =====================================================

imputer = SimpleImputer(strategy='mean')

df[['Age', 'Salary']] = imputer.fit_transform(
    df[['Age', 'Salary']]
)

print("\nAfter Handling Missing Values\n")
print(df)


# =====================================================
# STEP 3: ENCODE CATEGORICAL DATA
# =====================================================

encoder = LabelEncoder()

df['Gender'] = encoder.fit_transform(df['Gender'])
df['Purchased'] = encoder.fit_transform(df['Purchased'])

print("\nAfter Encoding\n")
print(df)


# =====================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# =====================================================

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# =====================================================
# STEP 5: STANDARDIZATION
# =====================================================

sc = StandardScaler()

X_standard = sc.fit_transform(X)

print("\nStandardized Data\n")
print(X_standard)


# =====================================================
# STEP 6: NORMALIZATION
# =====================================================

mm = MinMaxScaler()

X_normal = mm.fit_transform(X)

print("\nNormalized Data\n")
print(X_normal)


# =====================================================
# STEP 7: TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_normal,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))


# =====================================================
# STEP 8: CREATE MACHINE LEARNING MODEL
# =====================================================

model = KNeighborsClassifier(n_neighbors=1)


# =====================================================
# STEP 9: SEQUENTIAL FORWARD SELECTION (SFS)
# =====================================================

print("\n----------------------------------------")
print("Sequential Forward Selection (SFS)")
print("----------------------------------------")

# Start with an empty feature set
selected_features = []

# All available features
remaining_features = list(range(X_train.shape[1]))

# Select 2 important features
required_features = 2

while len(selected_features) < required_features:

    best_feature = None
    best_accuracy = -1

    # Try each remaining feature
    for feature in remaining_features:

        # Add the feature temporarily
        current_features = selected_features + [feature]

        # Train model
        model.fit(
            X_train[:, current_features],
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test[:, current_features]
        )

        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            "Trying feature:",
            X.columns[feature],
            "Accuracy:",
            accuracy
        )

        # Select feature with highest accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature = feature

    # Add best feature to selected set
    selected_features.append(best_feature)

    # Remove it from remaining features
    remaining_features.remove(best_feature)


print("\nSelected Features using SFS:")

for feature in selected_features:
    print(X.columns[feature])


# =====================================================
# STEP 10: SEQUENTIAL BACKWARD SELECTION (SBS)
# =====================================================

print("\n----------------------------------------")
print("Sequential Backward Selection (SBS)")
print("----------------------------------------")

# Start with all features
selected_features_sbs = list(range(X_train.shape[1]))

while len(selected_features_sbs) > required_features:

    feature_to_remove = None
    best_accuracy = -1

    # Try removing each feature
    for feature in selected_features_sbs:

        # Temporarily remove the feature
        current_features = [
            f for f in selected_features_sbs
            if f != feature
        ]

        # Train model
        model.fit(
            X_train[:, current_features],
            y_train
        )

        # Predict
        predictions = model.predict(
            X_test[:, current_features]
        )

        # Calculate accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            "Removing feature:",
            X.columns[feature],
            "Accuracy:",
            accuracy
        )

        # Select the removal that gives best performance
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            feature_to_remove = feature

    # Remove selected feature permanently
    selected_features_sbs.remove(feature_to_remove)


print("\nSelected Features using SBS:")

for feature in selected_features_sbs:
    print(X.columns[feature])


# =====================================================
# FINAL RESULT
# =====================================================

print("\n========================================")
print("FINAL RESULT")
print("========================================")

print("\nSFS Selected Features:")

for feature in selected_features:
    print("-", X.columns[feature])

print("\nSBS Selected Features:")

for feature in selected_features_sbs:
    print("-", X.columns[feature])
