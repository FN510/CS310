# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:28:03 2018

@author: Franklin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessingFunctions


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from collections import defaultdict
from subprocess import check_call

# tree module used to build decision tree classifier




maths = pd.read_csv('../data/student/student-mat.csv', sep =';' )
maths_copy = maths.copy()

feature_names = list(maths_copy)[:30] # all the attributes except grades (G1, G2, G3)

# preprocessing
# Encode G1, G2, G3 into classes: 4 (A), 3 (B), 2 (C), 1 (D), 0 (F)
preprocessingFunctions.encode_grades_to_5_class(maths_copy)

# Binary classification of grades
# encode_grades_to_pass_fail(maths_copy)


# encode all attributes but G1, G2, G3
d = defaultdict(preprocessing.LabelEncoder)
maths_copy.loc[:, feature_names] =  maths_copy.loc[:, feature_names].apply(
	lambda x: d[x.name].fit_transform(x)) # for each col, fit label-encoder and return encoded values

X = maths_copy[feature_names].values # data with only predictor attributes
y = maths_copy.iloc[:, 32].values # data with only target attribute (G3)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()