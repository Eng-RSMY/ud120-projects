#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

data_shrink_factor = 100
# features_train = features_train[:len(features_train)//data_shrink_factor]
# labels_train = labels_train[:len(labels_train)//data_shrink_factor]


for item in [10, 100, 100, 10000]:
    cvalue = item
    clf = SVC(kernel='rbf', C=cvalue)

    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    print("training svc took: {0:.2f} seconds".format(t1-t0))

    pred = clf.predict(features_test)
    t2 = time()
    print("predicting using svc took: {0:.2f} seconds".format(t2-t1))

    accu = accuracy_score(labels_test, pred)
    print("for C = {:.0f}, accuracy is {:.4f}".format(cvalue, accu))
    print("{:d} are predicted to be Chris's email".format(pred[pred == 1].sum()))




