import numpy as np
import pandas as pd

from sklearn import preprocessing
import preprocessingFunctions
from collections import defaultdict
from subprocess import check_call
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score


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

max_k = int(maths_copy.shape[0]*0.9)
max_score = 0.0
k_values = [i for i in range(1,max_k,2)]
acc_values = [0.0 for i in range(len(k_values))]
std_values = [0.0 for i in range(len(k_values))]
# train model for range of k values, store 10-fold cv mean
for i in range(len(k_values)):
    knn = KNeighborsClassifier(n_neighbors=k_values[i], weights= 'distance', metric='manhattan')
    scores = cross_val_score(knn, features, target, cv=10) # 10-fold cross validation
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