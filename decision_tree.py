import numpy as np
import pandas as pd
import preprocessingFunctions
from stats import mcNemar
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
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

bin = 0 # flag to indicate type of classification task: 1- binary, 0- 5-class
test_type = 'Binary classification'
if bin==1:

    # Binary classification: encode G3 to classes 0 (fail), 1 (pass)
    preprocessingFunctions.encode_grades_to_pass_fail(maths_copy)

else:
    test_type = '5-class classification'
    # Encode G3 into classes: 4 (A), 3 (B), 2 (C), 1 (D), 0 (F)
    preprocessingFunctions.encode_grades_to_5_class(maths_copy)

feature_names = list(maths_copy)[:30] # list of attribute names

# dataset split into features and targets
features = maths_copy[feature_names].values # data with only predictor attributes
target = maths_copy.values[:, 30] # (G3)

# initialise variables to store best hyperparameters
max_score = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
min_split = 0.0
min_leaf = 0.0
max_depth = 0.0

sel_method = 'entropy' # attribute measure
for i in range(10,100,10):
    for j in range(10, 100, 10):
        for k in range(1, 20):
            dtree = DecisionTreeClassifier(criterion = sel_method, random_state = 0, min_samples_split=i,
                min_samples_leaf=j, max_depth=k) # create decision tree with given hyperparameters
            scores = cross_val_score(dtree, features, target, cv=10) # perform 10-fold cross-validation
            if scores.mean()>max_score.mean(): # record highest mean accuracy
                max_score = scores
                min_split = i
                min_leaf = j
                max_depth = k
print 'Cross-validation accuracy scores\n:', max_score
print "Average: %0.2f (+/- %0.2f)" % (max_score.mean(), max_score.std())
print 'Best hyperparameter values:'
print 'min_samples_split: {}'.format(min_split)
print 'min_leaf: {}'.format(min_leaf)
print 'max_depth: {}'.format(max_depth)