import numpy as np
import pandas as pd

import preprocessingFunctions
from stats import mcNemar
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, RFE, f_classif,chi2, SelectFromModel

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from collections import defaultdict
from subprocess import check_call
import matplotlib.pyplot as plt

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

maths_copy_fs = maths_copy.sample(frac=0.1, random_state=1) # subset of data for feature selection
maths_copy_cv = maths_copy.drop(maths_copy_fs.index) # dataset for use with cross validation

# dataset for feature selection split into attributes and targets
features_fs = maths_copy_fs[feature_names].values
target_fs = maths_copy_fs.values[:, 30] # (G3)

# select best features
dtree = DecisionTreeClassifier(criterion='entropy').fit(features_fs, target_fs)
sorted_features = sorted(dtree.feature_importances_, reverse=True)[0:3] # sort attribute importances, keep top 3

best_features_index = list() # store index of best features
for i in range(len(sorted_features)):
    best_features_index.append(list(dtree.feature_importances_).index(sorted_features[i]))

best_features_names = list() # create list of top three features
print 'Top three features using sample data:'
for i in range(len(best_features_index)):
    best_features_names.append(feature_names[best_features_index[i]])
    print '{}. {}'.format(i+1, best_features_names[i])

# keep only top three attributes for cross-validation (cv)
#  split cv data into attributes and target
features_cv = maths_copy_cv[best_features_names].values
target_cv = maths_copy_cv.values[:, 30] # (G3)

max_k = int(target_cv.shape[0]*0.9)
max_score = 0.0
k_values = [i for i in range(1,max_k,2)]
acc_values = [0.0 for i in range(len(k_values))]
std_values = [0.0 for i in range(len(k_values))]
# train model for range of k values, store 10-fold cv mean
for i in range(len(k_values)):
    knn = KNeighborsClassifier(n_neighbors=k_values[i], weights= 'distance', metric='manhattan')
    scores = cross_val_score(knn, features_cv, target_cv, cv=10) # 10-fold cross validation
    acc_values[i] = round(scores.mean(),2)
    std_values[i] = round(scores.std(),4)
    if scores.mean()> max_score:
        max_score = scores.mean()

# create table for the results
k_graph = {
    'k value': pd.Series(k_values),
    'Accuracy': pd.Series(acc_values),
    'SD': pd.Series(std_values)
}
k_graph = pd.DataFrame(k_graph)

print test_type
print k_graph # print table
plt.plot(k_values, acc_values) # plot mean accuracy / k value
plt.ylabel('Mean accuracy')
plt.xlabel('Value of K')
plt.show()