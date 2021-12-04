import pandas as pd
import numpy as np
import random
import imblearn
from imblearn.over_sampling import SMOTE

# Read in the dataset
insTrain = pd.read_csv("ins_train.csv")
insTest = pd.read_csv("ins_test.csv")

# Summary of train datasets
head = insTrain.head()

# How many of each value
insuranceCount = insTrain.groupby('Response').count()
# Fairly imbalanced dataset with 62,601 with response 1 and 319,553 with response 0

# Create function and apply for random oversampling - binary
def overSamplingBinary(X, y, targetBalance = 0.5):
    # define majority/minority classes based on the inputted
    # target column (y)
    yMajority = y.value_counts().idxmax()
    yMinority = y.value_counts().idxmin()
    # We next create separate majority and minority dataframes which are converted
    # to lists
    X['target'] = y
    majority = X[X['target'] == yMajority].values.tolist()
    minority = X[X['target'] == yMinority].values.tolist()
    # Next we implement a while loop to keep randomly selecting rows
    # from the minority data until the target balance between minority
    # and majority is achieved
    enlargedMinority = []
    while len(enlargedMinority)/(len(majority) + len(enlargedMinority)) < targetBalance:
        randomValue = random.choice(minority)
        enlargedMinority.append(randomValue)
    # Take the original column names
    columnNames = list(X.columns)
    # Create a new dataset by comining target and features
    newDataset = enlargedMinority + majority
    # combine this back as a df with original column names
    # for features
    newDataset = pd.DataFrame(newDataset, columns = columnNames)
    return newDataset
# Take the features and target
featuresTrain = insTrain.drop('Response', axis=1)
targetTrain = insTrain['Response']
oversampledData = overSamplingBinary(featuresTrain, targetTrain)

# Create function for random undersampling - binary
def underSamplingBinary(X, y, targetBalance = 0.5):
    # define majority/minority classes
    yMajority = y.value_counts().idxmax()
    yMinority = y.value_counts().idxmin()
    # Create separate majority and minority lists
    X['target'] = y
    majority = X[X['target'] == yMajority].values.tolist()
    minority = X[X['target'] == yMinority].values.tolist()
    # While the length of the majority is larger than the targeted balance
    # we randomly remove one instance from the majority dataset
    while len(majority)/(len(minority) + len(majority)) > targetBalance:
        majority.pop(random.randrange(len(majority)))
    # Take the original column names
    columnNames = list(X.columns)
    newDataset = minority + majority
    newDataset = pd.DataFrame(newDataset, columns = columnNames)
    return newDataset
# Take the features and target
underSampledData = underSamplingBinary(featuresTrain, targetTrain)

# Smote
# All values need to be numeric so convert labels in dataset
insTrain['Gender'] = insTrain['Gender'].astype('category').cat.codes
insTrain['Vehicle_Age'] = insTrain['Vehicle_Age'].astype('category').cat.codes
insTrain['Vehicle_Damage'] = insTrain['Vehicle_Damage'].astype('category').cat.codes
featuresTrain = insTrain.drop('Response', axis=1)
targetTrain = insTrain['Response']

# Implement imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=1)
X_sm, y_sm = smote.fit_resample(featuresTrain, targetTrain)
print(y_sm.value_counts())


