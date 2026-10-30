# Experiment No. 5
# Implementation of Find-S Algorithm


# ---------------------------------
# Step 1: Create Training Dataset
# ---------------------------------

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]


# ---------------------------------
# Step 2: Define Attribute Names
# ---------------------------------

attributes = [
    'Sky',
    'Temperature',
    'Humidity',
    'Wind',
    'Water',
    'Forecast'
]


# ---------------------------------
# Step 3: Initialize Most Specific Hypothesis
# ---------------------------------

hypothesis = ['0'] * len(attributes)

print("Initial Hypothesis:")
print(hypothesis)


# ---------------------------------
# Step 4: Find-S Algorithm
# ---------------------------------

for row in data:

    # Check whether example is positive
    if row[-1] == 'Yes':

        # Compare each attribute
        for i in range(len(attributes)):

            # If hypothesis is empty
            if hypothesis[i] == '0':

                hypothesis[i] = row[i]

            # If values are different
            elif hypothesis[i] != row[i]:

                hypothesis[i] = '?'


        print("\nAfter processing positive example:")
        print(hypothesis)


# ---------------------------------
# Step 5: Display Final Hypothesis
# ---------------------------------

print("\nFinal Hypothesis:")
print(hypothesis)
