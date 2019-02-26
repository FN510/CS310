import numpy as np
import pandas as pd
import preprocessingFunctions
from stats import mcNemar
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from collections import defaultdict
from subprocess import check_call

# import csv file and create a copy
maths = pd.read_csv('../data/student/student-mat.csv', sep =';' )
maths_copy = maths.copy()

# preprocessing

maths_copy.drop(['G1', 'G2'], axis=1,inplace=True) # Remove attributes G1 and G2

# encode all attributes but G3
d = defaultdict(preprocessing.LabelEncoder)
maths_copy.iloc[:, : 30] =  maths_copy.iloc[:, : 30].apply(
	lambda x: d[x.name].fit_transform(x)) # for each col, fit label-encoder and return encoded values

# feature re-scaling: ->[0,1]
maths_copy = maths_copy - maths_copy.min()
maths_copy = maths_copy/maths_copy.max()-maths_copy.min()
maths_copy.values[:,30] = maths.values[:,30] # leave the target unchanged

bin = 0 # flag to indicate classification task
test_type = 'Binary classification'
if bin==1:

    # Binary classification: encode G3 to classes 0 (fail), 1 (pass)
    preprocessingFunctions.encode_grades_to_pass_fail(maths_copy)

else:
    test_type = '5-class classification'
    # Encode G3 into classes: 4 (A), 3 (B), 2 (C), 1 (D), 0 (F)
    preprocessingFunctions.encode_grades_to_5_class(maths_copy)

feature_names = list(maths_copy)[:30] # list of attribute names

# For feature selection models
# ----------------------------------------------------------
# best_features_names = [] # add best features as strings
# features_2 = maths_copy[best_features_names].values
# target_2 = maths_copy.values[:, 30] # (G3)
# ----------------------------------------------------------

# dataset split into features and targets
features = maths_copy[feature_names].values # data with only predictor attributes
target = maths_copy.values[:, 30] # (G3)


X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.1, random_state=1) # split data into training and test set (9:1 ratio as in tests)

# example: compare Decision trees using information gain and gin impurity - use optimal hyperparameters
dtree_ig = DecisionTreeClassifier(criterion='entropy', random_state = 0, min_samples_split=10,
                min_samples_leaf=10, max_depth=1)
dtree_gini = DecisionTreeClassifier(criterion='gini', random_state = 0, min_samples_split=10,
                min_samples_leaf=10, max_depth=1)

fitted_dtree_ig = dtree_ig.fit(X_train, y_train)
fitted_dtree_gini = dtree_gini.fit(X_train, y_train)
ig_pred = fitted_dtree_ig.predict(X_test)
gini_pred = fitted_dtree_gini.predict(X_test)

print test_type
print mcNemar(ig_pred, gini_pred, y_test) # apply McNemar's test to two classifiers
