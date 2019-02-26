import numpy as np
import pandas as pd
import preprocessingFunctions
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, RFE, RFECV, chi2
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
from collections import defaultdict
from subprocess import check_call

# tree module used to build decision tree classifier

maths = pd.read_csv('../data/student/student-mat.csv', sep =';' )
maths_copy = maths.copy()

# preprocessing
# -----------------
maths_copy.drop(['G1', 'G2'], axis=1,inplace=True) # Remove attributes G1 and G2

# Binary classification of grades
#preprocessingFunctions.encode_grades_to_pass_fail(maths_copy)
#
# encode all attributes but G3
d = defaultdict(preprocessing.LabelEncoder)
maths_copy.iloc[:, : 30] =  maths_copy.iloc[:, : 30].apply(
	lambda x: d[x.name].fit_transform(x)) # for each col, fit label-encoder and return encoded values

# Inverse the encoded
# =============================================================================
# maths_copy.iloc[:, : 30]  = maths_copy.iloc[:, : 30].apply(
# lambda x: d[x.name].inverse_transform(x))
# print maths_copy.iloc[:10, 0]
# =============================================================================

# feature re-scaling
maths_copy = maths_copy - maths_copy.min()
maths_copy = maths_copy/maths_copy.max()-maths_copy.min()
maths_copy.values[:,30] = maths.values[:,30] # leave the target unchanged

# Encode G3 into classes: 4 (A), 3 (B), 2 (C), 1 (D), 0 (F)
preprocessingFunctions.encode_grades_to_5_class(maths_copy)

maths_copy_fs = maths_copy.sample(frac=0.1, random_state=1) # subset of data for feature selection
maths_copy_cv = maths_copy.drop(maths_copy_fs.index) # dataset for use with cross validation

print 'cv data shape: {}'.format(maths_copy_cv.shape)
print 'fs data shape: {}'.format(maths_copy_fs.shape)

feature_names = list(maths_copy_cv)[:30]

# dataset for fs split into features and targets
features_fs = maths_copy_fs[feature_names].values # data with only predictor attributes
target_fs = maths_copy_fs.values[:, 30] # (G3)

# using the information gain attribute measure
sel_method = 'entropy'
dtree = DecisionTreeClassifier(
        criterion = sel_method, random_state = 0, min_samples_split=50,
        min_samples_leaf=60, max_depth=8)

# fitted_maths_dtree = dtree.fit(features, target)
# recursive feature selection (RFE)
rfecv = RFE(estimator=dtree, step=1, n_features_to_select=3)
rfecv.fit(features_fs, target_fs) # perform fs using allocated fs dataset
print 'Number of features selected {}'.format(rfecv.n_features_)
print rfecv.ranking_
selected_features = list()
for i in range(len(rfecv.ranking_)):
    if rfecv.ranking_[i]==1:
        selected_features.append(feature_names[i])

print selected_features

# dataset for cv split into features and targets
features_cv = maths_copy_cv[selected_features].values # data with only predictor attributes
target_cv = maths_copy_cv.values[:, 30] # (G3)

# scores = cross_val_score(dtree, features_cv, target_cv, cv=10)
# print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

# create graphical visualisation of dt
# tree.export_graphviz(fitted_maths_dtree, out_file = 'dtree.dot', feature_names = features_names, class_names = ['A', 'B', 'C', 'D', 'E', 'F'])