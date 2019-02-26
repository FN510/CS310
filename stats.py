import numpy as np
import math

# Calculate the contingency table for McNemar's test
def mcNemar(pred_A, pred_B, truth):
    # A and B hold predictions for a test swt
    # truth holds the ground truth
    incorrect_ab = 0
    incorrect_a = 0
    incorrect_b = 0
    correct_ab = 0

    for i in range(truth.size):
        if pred_A[i] != truth[i] and pred_B[i] != truth[i]:
            incorrect_ab += 1 # both A and B predict incorrectly
        elif pred_A[i] != truth[i] and pred_B[i] == truth[i]:
            incorrect_a += 1 # only b predicts correctly
        elif pred_A[i] == truth[i] and pred_B[i] != truth[i]:
            incorrect_b += 1 # only a predicts correctly
        else:
            correct_ab +=1
    if math.fabs(incorrect_a-incorrect_b) >0:
        if ((math.fabs(incorrect_a-incorrect_b)-1)**2)/(incorrect_a+incorrect_b) > 3.841459:
            print 'significant McNemar result'
        else:
            print  'McNemar result not significant'
    else:
        print  'McNemar result not significant'

    return np.array([incorrect_ab, incorrect_a, incorrect_b, correct_ab])