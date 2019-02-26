import numpy as np
import pandas as pd
import preprocessingFunctions

from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from collections import defaultdict

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
count = list([0 for i in range(len(feature_names))]) # store count for each attribute appearing in top three


for i in range(300): # 300 iterations of ranking based on random sample

    maths_copy_fs = maths_copy.sample(frac=0.1, random_state=1) #subset of data for feature selection
    maths_copy_cv = maths_copy.drop(maths_copy_fs.index) # dataset for use with cross validation

    # dataset for fs. split into features and targets
    features_fs = maths_copy_fs[feature_names].values
    target_fs = maths_copy_fs.values[:, 30] # (G3)

    # select best features
    dtree = DecisionTreeClassifier(criterion='entropy').fit(features_fs, target_fs)
    selected_features = list()
    sorted_features = sorted(dtree.feature_importances_, reverse=True)[0:3]
    # print sorted_features
    for j in range(len(sorted_features)):
        selected_features.append(list(dtree.feature_importances_).index(sorted_features[j]))

    best_features = list()

    for k in range(len(selected_features)):
        best_features.append(feature_names[selected_features[k]])
        count[selected_features[k]]+=1

for f in range(len(feature_names)):
    print '{}. {}, {}'.format(f+1, feature_names[f], count[f])