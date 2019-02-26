import numpy as np
import pandas as pd
import preprocessingFunctions
from stats import mcNemar
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, RFE, RFECV, chi2
from sklearn.feature_selection import f_classif
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

# dataset split into features and targets
features = maths_copy[feature_names].values # data with only predictor attributes
target = maths_copy.values[:, 30] # (G3)

# initialise variables to store best hyperparameters
max_score = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
min_split = 0.0
min_leaf = 0.0
max_depth = 0.0

# using the information gain attribute measure
sel_method = 'entropy'
# for i in range(10,100,10):
#     for j in range(10, 100, 10):
#         for k in range(1, 20):
#             dtree = DecisionTreeClassifier(criterion = sel_method, random_state = 0, min_samples_split=i,
#                 min_samples_leaf=j, max_depth=k)
#             scores = cross_val_score(dtree, features, target, cv=10)
#             if scores.mean()>max_score.mean():
#                 max_score = scores
#                 min_split = i
#                 min_leaf = j
#                 max_depth = k
# print max_score
# print "Accuracy: %0.2f (+/- %0.2f)" % (max_score.mean(), max_score.std())
# print 'min_samples_split: {}'.format(min_split)
# print 'min_leaf: {}'.format(min_leaf)
# print 'max_depth: {}'.format(max_depth)


# generate example decision tree
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.1, random_state=1)
dtree_ig = DecisionTreeClassifier(criterion='entropy', random_state = 0, min_samples_split=10,
                min_samples_leaf=10, max_depth=1)
dtree_gini = DecisionTreeClassifier(criterion='gini', random_state = 0, min_samples_split=10,
                min_samples_leaf=10, max_depth=1)

fitted_dtree_ig = dtree_ig.fit(X_train, y_train)
fitted_dtree_gini = dtree_gini.fit(X_train, y_train)
ig_pred = fitted_dtree_ig.predict(X_test)
gini_pred = fitted_dtree_gini.predict(X_test)

print confusion_matrix(y_test, ig_pred, labels=[4,3,2,1,0])
print y_test
print test_type
print 'Items in test set: {}'.format(y_test.size)
print mcNemar(ig_pred, gini_pred, y_test)


# fitted_maths_dtree = dtree.fit(X_train, y_train)
# print np.shape(np.where(y_train==0))
# create graphical visualisation of dt
tree.export_graphviz(fitted_dtree_ig, out_file = 'dtree.dot', feature_names = feature_names, class_names = ['F', 'D', 'C', 'B', 'A'])